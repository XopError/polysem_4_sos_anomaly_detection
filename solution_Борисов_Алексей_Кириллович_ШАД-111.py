import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = Path('data_train')
OUTPUT_DIR = Path('output')
PLOTS_DIR = OUTPUT_DIR / 'plots'

# Минимальный абсолютный порог OTS для рассмотрения как аномалии.
# Обоснование: средний Weight ~2000. Значение 5000 означает минимум 3 запроса.
# Это выполняет требование: "Малый OTS сам по себе не является аномалией".
MIN_OTS_THRESHOLD = 5000.0

def setup_directories():
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

def load_and_preprocess_data():
    """
    Загружает все parquet файлы из папок month=*, фильтрует по условиям задачи.
    Возвращает очищенный DataFrame.
    """
    print("Загрузка данных...")
    # Ищем все parquet файлы в подпапках data_train.
    parquet_files = list(DATA_DIR.glob('month=*/*.parquet'))
    if not parquet_files: raise FileNotFoundError(f"Не найдены файлы .parquet в директории {DATA_DIR}")

    # Только нужные колонки для экономии памяти.
    # ВАЖНО: В схеме parquet колонка называется 'CategoryNameDelivery'
    cols_to_keep = [
        'SubjectID', 'QueryText', 'BrandID', 'CategoryNameDelivery', 'Brand',
        'Weight', 'researchdate', 'BrandinDelivery',
        # Дополнительные колонки для аналитики
        'Пол', 'Возраст', 'Регион',
        'ResourceName', 'ResourceType', 'Platform', 'UseType',
        'Category1', 'Category2', 'Category3'
    ]

    dfs = []
    for file in parquet_files:
        df_chunk = pd.read_parquet(file, columns=cols_to_keep)
        dfs.append(df_chunk)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Всего загружено строк: {len(df):,}")

    # Приводим веса к float для корректных вычислений
    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')

    # Переименовываем колонку в соответствии с логикой задачи и выходными требованиями
    df = df.rename(columns={'CategoryNameDelivery': 'CategoryDelivery'})

    # Фильтрация согласно условию: BrandinDelivery == 1 и CategoryDelivery не пустой
    initial_len = len(df)
    df = df[(df['BrandinDelivery'] == 1) & (df['CategoryDelivery'].notna()) & (df['CategoryDelivery'] != '')]
    print(f"Строк после фильтрации (BrandinDelivery==1 и валидный CategoryDelivery): {len(df):,} (удалено {initial_len - len(df):,})")

    # Удаляем строки с нулевым или отрицательным весом, если такие вдруг есть
    df = df[df['Weight'] > 0]
    print(f"Строк после фильтрации по Weight > 0: {len(df):,}")

    return df

def detect_anomalies(df):
    """
    Агрегирует данные до уровня (SubjectID, researchdate, BrandID, CategoryDelivery)
    и применяет статистически обоснованный алгоритм поиска аномалий.
    """
    print("Агрегация данных и расчет daily_ots...")

    # Агрегация: считаем количество строк (запросов) и берем вес респондента за день
    df_agg = df.groupby(['SubjectID', 'researchdate', 'BrandID', 'CategoryDelivery', 'Brand']).agg(
        count_rows=('QueryText', 'count'),
        weight=('Weight', 'first') # Weight - это дневной вес, он одинаков для всех строк респондента в этот день
    ).reset_index()

    # Расчет daily_ots по формуле из условия
    df_agg['daily_ots'] = df_agg['weight'] * df_agg['count_rows']

    print("Расчет статистик и порогов для каждого бренда в каждый день...")
    group_cols = ['researchdate', 'CategoryDelivery', 'BrandID']

    # 1. Общий OTS бренда за день и доля респондента (интерпретируемый score)
    df_agg['brand_total_ots'] = df_agg.groupby(group_cols)['daily_ots'].transform('sum')
    df_agg['respondent_share'] = df_agg['daily_ots'] / df_agg['brand_total_ots']

    # 2. Количество респондентов в группе (для учета проблемы малых выборок)
    df_agg['group_count'] = df_agg.groupby(group_cols)['SubjectID'].transform('count')

    # 3. Статистики распределения
    df_agg['p99_ots'] = df_agg.groupby(group_cols)['daily_ots'].transform(lambda x: x.quantile(0.99) if len(x) > 0 else 0)
    df_agg['median_ots'] = df_agg.groupby(group_cols)['daily_ots'].transform('median')

    # 4. Расчет адаптивного порога
    # Базовый порог: максимум из 99-го перцентиля и (медиана * 5)
    base_threshold = np.maximum(df_agg['p99_ots'], df_agg['median_ots'] * 5)

    # Если выборка очень мала (< 3 респондентов), перцентили ненадежны.
    # Ужесточаем правило: требуем превышение медианы в 10 раз или абсолютный минимум.
    df_agg['threshold'] = np.where(
        df_agg['group_count'] < 3,
        np.maximum(df_agg['median_ots'] * 10, MIN_OTS_THRESHOLD),
        np.maximum.reduce([base_threshold, np.full(len(df_agg), MIN_OTS_THRESHOLD)])
    )

    # 5. Определение аномалий
    df_agg['is_anomaly'] = df_agg['daily_ots'] > df_agg['threshold']

    # Формирование человекочитаемой причины
    df_agg['reason'] = np.where(
        df_agg['is_anomaly'],
        df_agg['daily_ots'].round(2).astype(str) + ' > ' + df_agg['threshold'].round(2).astype(str) +
        f' (Доля в бренде: ' + (df_agg['respondent_share'] * 100).round(1).astype(str) + '%)',
        ''
    )

    # Формирование выходных таблиц
    anomalies_df = df_agg[df_agg['is_anomaly']][['SubjectID', 'researchdate']].drop_duplicates().reset_index(drop=True)

    reasons_df = df_agg[df_agg['is_anomaly']][[
        'SubjectID', 'researchdate', 'BrandID', 'Brand', 'CategoryDelivery',
        'daily_ots', 'respondent_share', 'threshold', 'reason'
    ]].copy()
    # Переименовываем respondent_share в score, как требуется в схеме, т.к. это численная причина (доля)
    reasons_df = reasons_df.rename(columns={'respondent_share': 'score'})

    print(f"Найдено уникальных аномальных пар (респондент-день): {len(anomalies_df):,}")
    print(f"Всего триггеров аномалий (респондент-день-бренд): {len(reasons_df):,}")

    return df_agg, anomalies_df, reasons_df

def apply_cleaning(df, anomalies_df):
    """
    Удаляет из исходного DataFrame все строки респондентов за те дни,
    когда они были признаны аномальными (согласно п.4 Условий).
    """
    print("Применение очистки данных...")
    # Создаем ключ для merge
    anomalies_df['merge_key'] = anomalies_df['SubjectID'].astype(str) + '_' + anomalies_df['researchdate'].astype(str)

    df_clean = df.copy()
    df_clean['merge_key'] = df_clean['SubjectID'].astype(str) + '_' + df_clean['researchdate'].astype(str)

    # Оставляем только те строки, которых НЕТ в списке аномалий
    df_clean = df_clean[~df_clean['merge_key'].isin(anomalies_df['merge_key'])].drop(columns='merge_key')

    print(f"Строк в исходных данных: {len(df):,}")
    print(f"Строк после очистки: {len(df_clean):,}")
    print(f"Удалено строк: {len(df) - len(df_clean):,} ({((len(df) - len(df_clean)) / len(df) * 100):.2f}%)")

    return df_clean

def plot_total_ots_before_after(df, df_clean, anomalies_df):
    """График изменения общего OTS до и после удаления аномалий по дням."""
    print("Построение total_ots_before_after.png...")

    # Считаем OTS на уровне респондент-день в исходных и очищенных данных
    # Чтобы не суммировать дубликаты, сначала агрегируем до SubjectID + researchdate
    df_daily = df.groupby(['researchdate', 'SubjectID'])['Weight'].first().reset_index()
    df_daily['ots'] = df_daily['Weight'] # Упрощенно: вес респондента как прокси дневной активности,
    # или точнее: сумма daily_ots по всем брендам. Сделаем точнее:

    df_agg_orig = df.groupby(['researchdate', 'SubjectID', 'BrandID', 'CategoryDelivery']).size().reset_index(name='counts')
    df_agg_orig = df_agg_orig.merge(df[['SubjectID', 'researchdate', 'Weight']].drop_duplicates(), on=['SubjectID', 'researchdate'])
    df_agg_orig['daily_ots'] = df_agg_orig['Weight'] * df_agg_orig['counts']
    total_ots_before = df_agg_orig.groupby('researchdate')['daily_ots'].sum().reset_index(name='total_ots')

    df_clean_agg = df_clean.groupby(['researchdate', 'SubjectID', 'BrandID', 'CategoryDelivery']).size().reset_index(name='counts')
    df_clean_agg = df_clean_agg.merge(df_clean[['SubjectID', 'researchdate', 'Weight']].drop_duplicates(), on=['SubjectID', 'researchdate'])
    df_clean_agg['daily_ots'] = df_clean_agg['Weight'] * df_clean_agg['counts']
    total_ots_after = df_clean_agg.groupby('researchdate')['daily_ots'].sum().reset_index(name='total_ots')

    merged = total_ots_before.merge(total_ots_after, on='researchdate', suffixes=('_before', '_after'))

    plt.figure(figsize=(14, 6))
    plt.plot(merged['researchdate'], merged['total_ots_before'], label='До очистки', marker='o', color='red', alpha=0.7)
    plt.plot(merged['researchdate'], merged['total_ots_after'], label='После очистки', marker='s', color='green', alpha=0.7)

    plt.title('Изменение общего дневного OTS до и после очистки от аномалий')
    plt.xlabel('Дата (researchdate)')
    plt.ylabel('Суммарный OTS')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'total_ots_before_after.png', dpi=300)
    plt.close()

def plot_daily_anomaly_count(anomalies_df):
    """Гистограмма количества аномальных респондентов по дням."""
    print("Построение daily_anomaly_count.png...")

    counts = anomalies_df.groupby('researchdate').size().reset_index(name='anomaly_count')

    plt.figure(figsize=(16, 7))
    sns.barplot(data=counts, x='researchdate', y='anomaly_count', color='skyblue')

    plt.title('Количество аномальных респондентов по дням')
    plt.xlabel('Дата')
    plt.ylabel('Количество аномальных респондентов')

    plt.xticks(rotation=45, ha='right')
    # Показывать только каждую N-ную дату для читаемости
    if len(counts) > 50:
        step = len(counts) // 20  # Показывать около 20 дат
        labels = plt.gca().get_xticklabels()
        for i, label in enumerate(labels):
            if i % step != 0: label.set_visible(False)

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'daily_anomaly_count.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_category_ots_change(df, df_clean):
    """
    Гистограмма изменения OTS по CategoryDelivery в процентах.
    """
    print("Построение category_ots_change.png...")

    # Расчет OTS до очистки по категориям
    df_agg_orig = df.groupby(['SubjectID', 'researchdate', 'CategoryDelivery', 'BrandID']).size().reset_index(name='counts')
    df_agg_orig = df_agg_orig.merge(df[['SubjectID', 'researchdate', 'Weight']].drop_duplicates(), on=['SubjectID', 'researchdate'])
    df_agg_orig['daily_ots'] = df_agg_orig['Weight'] * df_agg_orig['counts']
    ots_before_cat = df_agg_orig.groupby('CategoryDelivery')['daily_ots'].sum().reset_index(name='ots_before')

    # Расчет OTS после очистки по категориям
    df_clean_agg = df_clean.groupby(['SubjectID', 'researchdate', 'CategoryDelivery', 'BrandID']).size().reset_index(name='counts')
    df_clean_agg = df_clean_agg.merge(df_clean[['SubjectID', 'researchdate', 'Weight']].drop_duplicates(), on=['SubjectID', 'researchdate'])
    df_clean_agg['daily_ots'] = df_clean_agg['Weight'] * df_clean_agg['counts']
    ots_after_cat = df_clean_agg.groupby('CategoryDelivery')['daily_ots'].sum().reset_index(name='ots_after')

    merged = ots_before_cat.merge(ots_after_cat, on='CategoryDelivery')
    merged['change_percent'] = ((merged['ots_after'] - merged['ots_before']) / merged['ots_before'] * 100).round(2)
    merged = merged.sort_values('change_percent')

    plt.figure(figsize=(12, 8))
    colors = ['red' if x < 0 else 'green' for x in merged['change_percent']]
    sns.barplot(data=merged, x='change_percent', y='CategoryDelivery', palette=colors)

    plt.title('Изменение суммарного OTS по категориям (CategoryDelivery) после очистки, %')
    plt.xlabel('Процент изменения OTS')
    plt.ylabel('Категория (CategoryDelivery)')

    # Добавляем подписи значений на бары
    for i, v in enumerate(merged['change_percent']):
        plt.text(v + (1 if v >= 0 else -1.5), i, f"{v}%", va='center', fontweight='bold')

    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'category_ots_change.png', dpi=300)
    plt.close()
    print("Все обязательные графики сохранены.")

# Аналитические возможности
class AnalyticalTools:
    """Класс с функциями для дополнительного аналитического разреза данных."""

    def __init__(self, df_original, df_cleaned, anomalies_df):
        self.df_original = df_original
        self.df_cleaned = df_cleaned
        self.anomalies_df = anomalies_df
        self.additional_dir = OUTPUT_DIR / 'additionally'
        self.additional_dir.mkdir(exist_ok=True)

    def plot_respondent_characteristics(self):
        print("Создание графиков по характеристикам респондентов...")

        characteristics = ['Пол', 'Возраст', 'Регион']

        for col in characteristics:
            if col not in self.df_original.columns:
                print(f"  Колонка {col} не найдена, пропускаем...")
                continue

            orig_counts = self.df_original[col].value_counts(normalize=True) * 100
            clean_counts = self.df_cleaned[col].value_counts(normalize=True) * 100

            # Создание DataFrame для сравнения
            df_plot = pd.DataFrame({
                'До очистки': orig_counts,
                'После очистки': clean_counts
            }).fillna(0).sort_index()

            # Построение графика
            plt.figure(figsize=(12, 6))
            df_plot.plot(kind='bar', color=['lightcoral', 'lightgreen'], width=0.8)
            plt.title(f'Распределение по признаку "{col}" до и после очистки', fontsize=14, fontweight='bold')
            plt.ylabel('Процент респондентов (%)', fontsize=12)
            plt.xlabel(col, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(loc='best')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()

            plt.savefig(self.additional_dir / f'respondent_{col}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Сохранен график: respondent_{col}.png")

    def plot_resource_characteristics(self):
        print("Создание графиков по характеристикам ресурсов...")

        characteristics = ['ResourceName', 'ResourceType', 'Platform', 'UseType']

        for col in characteristics:
            if col not in self.df_original.columns:
                print(f"  Колонка {col} не найдена, пропускаем...")
                continue

            orig_counts = self.df_original[col].value_counts(normalize=True) * 100
            clean_counts = self.df_cleaned[col].value_counts(normalize=True) * 100

            # Создание DataFrame для сравнения
            df_plot = pd.DataFrame({
                'До очистки': orig_counts,
                'После очистки': clean_counts
            }).fillna(0).sort_index()

            fig, ax = plt.subplots(figsize=(16, 7))
            df_plot.plot(kind='bar', color=['lightcoral', 'lightgreen'], width=0.8, ax=ax)
            ax.set_title(f'Распределение по ресурсу "{col}" до и после очистки', fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('Процент запросов (%)', fontsize=12)
            ax.set_xlabel(col, fontsize=12)
            ax.legend(loc='best')
            ax.grid(axis='y', alpha=0.3)

            # Для ResourceName показываем подписи только для топ-10
            if col == 'ResourceName':
                # Находим топ-10 ресурсов по среднему проценту (до + после)
                df_plot['avg'] = (df_plot['До очистки'] + df_plot['После очистки']) / 2
                top_10 = df_plot.nlargest(10, 'avg').index

                # Получаем позиции всех столбцов
                x_positions = range(len(df_plot))

                # Показываем подписи только для топ-10
                labels = [label if label in top_10 else '' for label in df_plot.index]
                ax.set_xticks(x_positions)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

                # Добавляем аннотацию о том, что показаны только топ-10
                ax.text(0.99, 0.99, f'Показаны подписи для топ-10 из {len(df_plot)} ресурсов',
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    horizontalalignment='right', style='italic',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                # Для других характеристик показываем все подписи
                ax.tick_params(axis='x', rotation=45, labelsize=9)

            plt.subplots_adjust(bottom=0.25)

            plt.savefig(self.additional_dir / f'resource_{col}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Сохранен график: resource_{col}.png")

    def plot_category_characteristics(self):
        print("Создание графиков по уровням категорий...")

        categories = ['CategoryDelivery', 'Category1', 'Category2', 'Category3']

        for col in categories:
            if col not in self.df_original.columns:
                print(f"  Колонка {col} не найдена, пропускаем...")
                continue

            orig_counts = self.df_original[col].value_counts(normalize=True) * 100
            clean_counts = self.df_cleaned[col].value_counts(normalize=True) * 100

            df_plot = pd.DataFrame({
                'До очистки': orig_counts,
                'После очистки': clean_counts
            }).fillna(0).sort_index()

            fig, ax = plt.subplots(figsize=(14, 7))
            df_plot.plot(kind='bar', color=['lightcoral', 'lightgreen'], width=0.8, ax=ax)
            ax.set_title(f'Распределение по категории "{col}" до и после очистки', fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('Процент запросов (%)', fontsize=12)
            ax.set_xlabel(col, fontsize=12)
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            ax.legend(loc='best')
            ax.grid(axis='y', alpha=0.3)
            plt.subplots_adjust(bottom=0.25)

            plt.savefig(self.additional_dir / f'category_{col}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Сохранен график: category_{col}.png")

    def save_anomalous_queries(self, subject_id=None, research_date=None):
        """
        Сохраняет таблицу поисковых запросов для аномального респондента.
        Если параметры не указаны, берет первого аномального респондента.
        """
        print("Сохранение таблицы запросов аномального респондента...")

        if subject_id is None or research_date is None:
            if not self.anomalies_df.empty:
                subject_id = self.anomalies_df.iloc[0]['SubjectID']
                research_date = self.anomalies_df.iloc[0]['researchdate']
                print(f"  Выбран первый аномальный респондент: {subject_id} за {research_date}")
            else:
                print("  Нет аномальных респондентов для анализа")
                return

        # Получаем запросы из оригинального датасета
        queries_df = self.df_original[
            (self.df_original['SubjectID'] == subject_id) &
            (self.df_original['researchdate'] == research_date)
        ][['researchdate', 'SubjectID', 'Brand', 'CategoryDelivery', 'QueryText', 'Weight']].copy()

        if queries_df.empty:
            print(f"  Нет данных для SubjectID={subject_id} на дату {research_date}")
            return

        filename = f'anomalous_queries_{subject_id}_{research_date}.csv'
        queries_df.to_csv(self.additional_dir / filename, index=False, encoding='utf-8-sig')
        print(f"  Сохранена таблица запросов: {filename}")
        print(f"  Всего запросов: {len(queries_df)}")

    def plot_brand_ots_timeline(self, brand_id=None, top_n=1):
        """
        График изменения OTS по дням для выбранного бренда.
        Если brand_id не указан, берет бренд с наибольшим количеством аномалий.
        """
        print("Создание графика OTS для бренда...")

        if brand_id is None:
            # Находим бренд с наибольшим количеством аномалий
            if not self.anomalies_df.empty:
                # Получаем BrandID из anomaly_reasons (нужно будет передать или загрузить)
                print("  Укажите brand_id для построения графика")
                return
            else:
                print("  Нет данных для анализа")
                return

        # Агрегация OTS до очистки по дням для бренда
        df_agg_orig = self.df_original.groupby(['researchdate', 'SubjectID', 'BrandID']).size().reset_index(name='counts')
        df_agg_orig = df_agg_orig.merge(
            self.df_original[['SubjectID', 'researchdate', 'Weight']].drop_duplicates(),
            on=['SubjectID', 'researchdate']
        )
        df_agg_orig['daily_ots'] = df_agg_orig['Weight'] * df_agg_orig['counts']
        ots_before = df_agg_orig[df_agg_orig['BrandID'] == brand_id].groupby('researchdate')['daily_ots'].sum().reset_index(name='ots_before')

        # Агрегация OTS после очистки
        df_agg_clean = self.df_cleaned.groupby(['researchdate', 'SubjectID', 'BrandID']).size().reset_index(name='counts')
        df_agg_clean = df_agg_clean.merge(
            self.df_cleaned[['SubjectID', 'researchdate', 'Weight']].drop_duplicates(),
            on=['SubjectID', 'researchdate']
        )
        df_agg_clean['daily_ots'] = df_agg_clean['Weight'] * df_agg_clean['counts']
        ots_after = df_agg_clean[df_agg_clean['BrandID'] == brand_id].groupby('researchdate')['daily_ots'].sum().reset_index(name='ots_after')

        # Объединение
        merged = ots_before.merge(ots_after, on='researchdate', how='outer').fillna(0)

        if merged.empty:
            print(f"  Нет данных для бренда {brand_id}")
            return

        # Построение графика
        plt.figure(figsize=(14, 7))
        plt.plot(merged['researchdate'], merged['ots_before'], label=f'Бренд {brand_id} (До очистки)',
                marker='o', color='red', alpha=0.7, linewidth=2)
        plt.plot(merged['researchdate'], merged['ots_after'], label=f'Бренд {brand_id} (После очистки)',
                marker='s', color='green', alpha=0.7, linewidth=2)

        plt.title(f'Динамика OTS бренда {brand_id} до и после очистки', fontsize=14, fontweight='bold')
        plt.xlabel('Дата', fontsize=12)
        plt.ylabel('Суммарный OTS бренда', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Сохранение
        filename = f'brand_{brand_id}_ots_timeline.png'
        plt.savefig(self.additional_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Сохранен график: {filename}")

    def run_all_analytics(self, brand_id=None):
        print("\n" + "-"*60)
        print("Запуск аналитичеких возможностей (п. 8.2)")
        print("-"*60)

        self.plot_respondent_characteristics()
        self.plot_resource_characteristics()
        self.plot_category_characteristics()
        self.save_anomalous_queries()

        if brand_id: self.plot_brand_ots_timeline(brand_id)

        print("-"*60)
        print(f"Все дополнительные материалы сохранены в: {self.additional_dir.absolute()}")
        print("-"*60)

def main():
    print("-"*60)
    print("Запуск алгоритма поиска аномалий SoS")
    print("-"*60)

    setup_directories()

    df = load_and_preprocess_data()

    df_agg, anomalies_df, reasons_df = detect_anomalies(df)

    print("Сохранение anomalies.csv и anomaly_reasons.csv...")
    anomalies_df.to_csv(OUTPUT_DIR / 'anomalies.csv', index=False)
    reasons_df.to_csv(OUTPUT_DIR / 'anomaly_reasons.csv', index=False)

    df_clean = apply_cleaning(df, anomalies_df)

    plot_total_ots_before_after(df, df_clean, anomalies_df)
    plot_daily_anomaly_count(anomalies_df)
    plot_category_ots_change(df, df_clean)

    print("-"*60)
    print("Работа завершена успешно")
    print(f"Результаты сохранены в папке: {OUTPUT_DIR.absolute()}")
    print("-"*60)

    # Запуск аналитических возможностей
    tools = AnalyticalTools(df, df_clean, anomalies_df)

    # Пример: можно указать конкретный brand_id
    brand_id_example = reasons_df['BrandID'].iloc[0] if not reasons_df.empty else None

    tools.run_all_analytics(brand_id=brand_id_example)

if __name__ == "__main__":
    main()
