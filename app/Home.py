import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
# import spacy
from collections import namedtuple
from pathlib import Path
from scipy.stats import ttest_ind


# Set configuration
st.set_page_config(
    page_title="Web Presentation",
    page_icon='🖥️',
    layout='wide',
    initial_sidebar_state='expanded',
    menu_items={
        'About': '''This is developed by Daryl Ku (Research Associate), Academic Quality Unit,
        Office of Strategic Planning and Academic Quality (SPAQ), National Institute of
        Education, Singapore.''',
        'Report a Bug': None,
        'Get help': None
        }
    )

# Display title
st.title(
    '''Survey Analysis Prototype'''
    )

# Display instructions
st.markdown(
    '''
    This is a prototype to demonstrate the deployment of a data processing pipeline online. Data from 
    separate Excel files are read in individually. It is then wrangled and analysed. The output is a
    visualisation similar to that illustrated in its Power BI counterpart.
    Significance and effect size is always in relation to the previous year.
    ''')

# Create dictionary of the dimensions and their respective number of questions
dimensions = {
    'A': np.arange(1, 4),
    'B': np.arange(1, 8),
    'C': np.arange(1, 5),
    'D': np.arange(1, 5),
    'E': np.arange(1, 5),
    'F': np.arange(1, 4),
    'G': np.arange(1, 5),
    'H': np.arange(1, 4),
    'I': np.arange(1, 5),
    'J': np.arange(1, 5),
    'K': np.arange(1, 3),
    'L': np.arange(1, 5),
    'M': np.arange(1, 4),
    'N': np.arange(1, 4),
    'O': np.arange(1, 4),
    'P': np.arange(1, 4),
    }

# Years to simulate
survey_years = [
    2018,
    2019,
    2020,
    2021,
    2022,
    ]

# Expand the questions as 'A.1', 'A.2', 'B.1' ... etc
dimensions_questions = [f'{dimension}.{item}' for dimension, items in dimensions.items() for item in items]

# Set likert scale items and weights
likert_scale_items = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Somewhat Disagree': 3,
    'Somewhat Agree': 4,
    'Agree': 5,
    'Strongly Agree': 6,
    }

# Set sex items
sex_items = [
    'Female',
    'Male',
    ]

# Set program items
program_items = [
    'Degree (BA)',
    'Degree (BSc)',
    'Diploma in Education',
    'Diploma in Special Education',
    'PGDE (JC)',
    'PGDE (Pri)',
    'PGDE (Sec)',
    ]

if 'D:' in str(Path.cwd()):
    data_path = Path.cwd() / 'Data/'
else:
    data_path = Path.cwd() / 'app/Data/'

# @st.cache(allow_output_mutation=True)
# def getNLP():
#     return spacy.load( data_path / 'trained-pipeline-2000')

@st.cache(allow_output_mutation=True)
def get_dataframe():    

    # Read data files
    df_dict = {}

    for year in survey_years:
        # Construct pandas dataframe from Excel data
        df = pd.read_excel(
            f'{data_path}/Simulated_Data_{year}.xlsx',
            index_col=None,
            engine='openpyxl',
        )

        # Drop rows with NA in specified columns
        df = df.dropna(subset=dimensions_questions)

        # Construct a dict of column-value replacer to map Likert scale items to numeric values
        replacer = {question:likert_scale_items for question in dimensions_questions}
        P_2_replacer = {'P.2': {key: 7 - value for key, value in likert_scale_items.items()}}  # to invert P.2
        replacer = replacer | P_2_replacer

        # Replace values in dataframe using replacer
        df = df.replace(replacer)

        # Create a mean column for each dimension
        for dimension in dimensions:
            # Get the column names for this dimension
            columns = [question for question in dimensions_questions if question.startswith(f'{dimension}.')]

            # Get mean of the columns in this dimension
            df[dimension] = df[columns].mean(axis=1).round(2)

        # Add a year column to dataframe
        df['Year'] = year

        # Drop individual questions columns in the dimensions
        df = df.drop(columns=dimensions_questions)

        # Add dataframe to collection
        df_dict[year] = df

    # Concat all dataframes in df_dict
    df = pd.concat(df_dict.values(), axis=0, ignore_index=True)

    # Unpivot dimension columns
    columns_to_unpivot = set(df.columns) - set(dimensions.keys())
    df = df.melt(id_vars=columns_to_unpivot, value_vars=dimensions.keys(), var_name='Dimension', value_name='Dimension_Mean')

    return df

df = get_dataframe()

# st.dataframe(data=df, width=500, height=300, use_container_width=True)


selected_year_widget = st.sidebar.radio('year', survey_years[::-1], horizontal=True, key='selected_year_widget')
selected_programs_widget = st.sidebar.multiselect('programs', program_items, default=program_items, key='selected_programs_widget')
selected_sex_widget = st.sidebar.multiselect('sex', sex_items, default=sex_items, key='selected_sex_widget')

selected_year = selected_year_widget
selected_year_previous = x if (x := selected_year - 1) in df['Year'].values else selected_year
selected_programs = selected_programs_widget if len(selected_programs_widget) > 0 else program_items
selected_sex = selected_sex_widget if len(selected_sex_widget) > 0 else sex_items

def cohen_d(*, mean1=1, mean2=1, std1=1, std2=1):
    # Calculate Cohen's d using Welch’s t-test formula where variance of both groups are not equal
    diff = abs(mean1 - mean2)
    pooledstd = math.sqrt((std1**2 + std2**2) / 2)
    cohen_d_numeric = round(diff / pooledstd, 3)

    # Convert numeric d to categorical d
    match cohen_d_numeric:
        # Sawilowsky's rule of thumb
        case cohen_d_numeric if cohen_d_numeric >= 2.0:
            cohen_d_categorical = 'Huge'
        case cohen_d_numeric if cohen_d_numeric >= 1.2:
            cohen_d_categorical = 'Very Large'
        case cohen_d_numeric if cohen_d_numeric >= 0.8:
            cohen_d_categorical = 'Large'
        case cohen_d_numeric if cohen_d_numeric >= 0.5:
            cohen_d_categorical = 'Medium'
        case cohen_d_numeric if cohen_d_numeric >= 0.2:
            cohen_d_categorical = 'Small'
        case _:
            cohen_d_categorical = 'Very Small'

    Cohen_D = namedtuple('Cohen_D', ['numeric', 'categorical'])
    return Cohen_D(cohen_d_numeric, cohen_d_categorical)


def get_random_likert_scale_items() -> list:
    return rng.choice(
        [*likert_scale_items.keys()],
        size=respondentsCount,
        replace=True,
        p=[0.10, 0.10, 0.10, 0.20, 0.25, 0.25]  # 'Strongly Disagree', 'Disagree', 'Somewhat Disagree', 'Somewhat Agree', 'Agree', 'Strongly Agree'
        ).tolist()


def get_random_sex_items() -> list:
    return rng.choice(
        sex_items,
        size=respondentsCount,
        replace=True,
        p=[0.65, 0.35],  # 'Female', 'Male'
        ).tolist()


def get_random_program_items() -> list:
    return rng.choice(
        program_items,
        size=respondentsCount,
        replace=True,
        p=None,  # 'Degree (BA)', 'Degree (BSc)', 'Diploma in Education', 'Diploma in Special Education', 'PGDE (JC)', 'PGDE (Pri)', 'PGDE (Sec)'
        ).tolist()


def get_mean():
    mean_dict = {}

    for dimension in dimensions.keys():
        filters = ((df['Program'].isin(selected_programs)) &
                   (df['Sex'].isin(selected_sex)) &
                   (df['Dimension'] == dimension)
                  )
        the_mean = df[(df['Year'] == selected_year) & filters]['Dimension_Mean'].mean()

        mean_dict[dimension] = the_mean

    return mean_dict


def get_current_previous_year_P():
    p_value_dict = {}

    for dimension in dimensions.keys():
        if selected_year == selected_year_previous:
            p_value_dict[dimension] = 0
        else:
            filters = ((df['Program'].isin(selected_programs)) &
                       (df['Sex'].isin(selected_sex)) &
                       (df['Dimension'] == dimension)
                      )
            A = df[(df['Year'] == selected_year) & filters]['Dimension_Mean']
            B = df[(df['Year'] == selected_year_previous) & filters]['Dimension_Mean']

            p_value_dict[dimension] = ttest_ind(A, B).pvalue

    return p_value_dict


def get_current_previous_year_D():
    d_value_dict = {}

    for dimension in dimensions.keys():
        if selected_year == selected_year_previous:
            d_value_dict[dimension] = 0
        else:
            filters = ((df['Program'].isin(selected_programs)) &
                       (df['Sex'].isin(selected_sex)) &
                       (df['Dimension'] == dimension)
                      )
            A = df[(df['Year'] == selected_year) & filters]['Dimension_Mean']
            B = df[(df['Year'] == selected_year_previous) & filters]['Dimension_Mean']

            d_value_dict[dimension] = cohen_d(mean1=A.mean(), mean2=B.mean(), std1=A.std(), std2=B.std()).numeric

    return d_value_dict

filters = ((df['Year'] == selected_year) &
           (df['Program'].isin(selected_programs)) &
           (df['Sex'].isin(selected_sex))
           )

filtered_df = df[filters]
filtered_df = filtered_df.drop_duplicates(subset=['Dimension'])

mean_values = get_mean()
filtered_df['Mean'] = filtered_df['Dimension'].apply(lambda p: mean_values[p])

p_values = get_current_previous_year_P()
filtered_df['Significance'] = filtered_df['Dimension'].apply(lambda p: p_values[p])

d_values = get_current_previous_year_D()
filtered_df['Effect Size'] = filtered_df['Dimension'].apply(lambda d: d_values[d])

# st.dataframe(data=filtered_df, width=500, height=300, use_container_width=True)

# Construct a bar trace for pvalue
mean_trace = go.Bar(
    name='Mean',
    x=filtered_df['Dimension'],
    y=filtered_df['Mean'].round(2),
    # customdata=results_df[['Effect Size Numerical', 'Effect Size Categorical']],
    # hovertemplate=(
    #     'P Value: %{y}<br>' +
    #     'Effect Size Numerical: %{customdata[0]}<br>' +
    #     'Effect Size Categorical: %{customdata[1]}<br>' +
    #     '<extra></extra>'  # remove the trace name
    #     ),
    text=filtered_df['Mean'].round(2), # data label
    textangle=0,
    textposition ='inside',
    )

# Construct a bar trace for effect size
# effect_size_trace = go.Bar(
#     name='Effect Size',
#     x=results_df['2021 → 2022 Dimension'],
#     y=results_df['Effect Size Numerical'],
#     )

# Create figure object using constructed traces
fig1 = go.Figure(
    data=[mean_trace],
    )

fig1.update_layout(
    title={
        'text': 'Mean',
        'x':0.5,
        'y':0.9,
        'xanchor': 'auto',
        'yanchor': 'auto',
    },
    xaxis_title=f'Mean for each Dimension in {selected_year}',
    yaxis_title='Dimension_Mean',    
    template='plotly_dark',
    height=600,
    )
st.plotly_chart(fig1, use_container_width=True)



# Construct a bar trace for significance
significance_trace = go.Bar(
    name='Significance',
    x=filtered_df['Dimension'],
    y=filtered_df['Significance'].round(2),
    # customdata=results_df[['Effect Size Numerical', 'Effect Size Categorical']],
    # hovertemplate=(
    #     'P Value: %{y}<br>' +
    #     'Effect Size Numerical: %{customdata[0]}<br>' +
    #     'Effect Size Categorical: %{customdata[1]}<br>' +
    #     '<extra></extra>'  # remove the trace name
    #     ),
    text=filtered_df['Significance'].round(2), # data label
    textangle=0,
    textposition ='inside',
    )

# Construct a bar trace for effect size
effect_size_trace = go.Bar(
    name='Effect Size',
    x=filtered_df['Dimension'],
    y=filtered_df['Effect Size'].round(2),
    # customdata=results_df[['Effect Size Numerical', 'Effect Size Categorical']],
    # hovertemplate=(
    #     'P Value: %{y}<br>' +
    #     'Effect Size Numerical: %{customdata[0]}<br>' +
    #     'Effect Size Categorical: %{customdata[1]}<br>' +
    #     '<extra></extra>'  # remove the trace name
    #     ),
    text=filtered_df['Effect Size'].round(2), # data label
    textangle=0,
    textposition ='inside',
    )

# Create figure object using constructed traces
fig2 = go.Figure(
    data=[significance_trace, effect_size_trace],
    )

fig2.update_layout(
    title={
        'text': 'Significance and Effect Size',
        'x':0.5,
        'y':0.9,
        'xanchor': 'auto',
        'yanchor': 'auto',
    },
    xaxis_title=f'Significance and Effect Size for each Dimension in {selected_year}',
    yaxis_title=None,
    yaxis_rangemode='nonnegative',
    height=600,
    template='plotly_dark',
    )

st.plotly_chart(fig2, use_container_width=True)










# nlp = getNLP()
df_test = pd.read_csv(
            f"{data_path / 'test_cleaned_w_predictions-10000.csv'}",
            index_col=None,
        )
df_test = df_test[['Label', 'News', 'News_Cat_Predicted', 'Accuracy']]
labels = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
# news_selection_items = []
# rng = np.random.default_rng()
# if 'df_test_random_rows' not in st.session_state:
#     st.session_state['df_test_random_rows'] = rng.choice(np.arange(0, len(df_test) / 4, dtype='int'), size=3, replace=False)
# for label in labels.values():
#     news_selection_items.extend(df_test[df_test['Label'] == label].iloc[st.session_state['df_test_random_rows']]['News'].tolist())
# selected_news_widget = st.selectbox('Select sample news', news_selection_items, key='selected_news_widget')
# selected_news_widget_text_area = st.text_area('Input news', value=selected_news_widget, height=100)
# predict_button_widget = st.button('Predict News Category')
# if predict_button_widget:
#     st.text(selected_news_widget_text_area)

df_test = df_test.rename(columns={'Label': 'Label', 'News': 'News', 'News_Cat_Predicted': 'Predicted' , 'Accuracy': 'Prediction Correct'})
st.dataframe(df_test, use_container_width=True)

overall_accuracy = round(df_test['Prediction Correct'].sum() / len(df_test), 3)
category_accuracy = {}
for label in labels.values():
    category_accuracy[label] = round(df_test[df_test['Label'] == label]['Prediction Correct'].sum() / len(df_test[df_test['Label'] == label]), 3)

results = f'''            
            Overall Accuracy: {overall_accuracy}
            "World" Accuracy: {category_accuracy['World']}
            "Sports" Accuracy: {category_accuracy['Sports']}
            "Business" Accuracy: {category_accuracy['Business']}
            "Sci/Tech" Accuracy: {category_accuracy['Sci/Tech']}
            '''
st.text(results)
