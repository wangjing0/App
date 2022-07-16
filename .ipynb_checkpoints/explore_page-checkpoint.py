import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#from shapely.geometry import Point
#import geopandas as gpd
#from geopandas import GeoDataFrame
from PIL import Image

@st.cache
def load_data():
    df = pd.read_pickle("./models/df_proper_cleaned.pkl")
    return df

df = load_data()

def show_explore_page():
    width = 800
    st.title("Activities, data and insights")

    st.write(
    """
    ###Geographic distribution of our users over the world
    """
    )
    
    image = Image.open('./images/geo_density_.png')
    #st.image(image, caption="Geographic distribution of our users over the world ",use_column_width=True)
    
    st.write(
    """
    ###Heart rate history of individual user
    """
    )
    image = Image.open('./images/hr_history.png')
    #st.image(image, caption="Heart rate history of individual user.",use_column_width=True)
    
    st.write(
    """
    ###Training Route example
    """
    )
    image = Image.open('./images/route_run.png')
    #st.image(image, caption="Training Route",use_column_width=True)
    
    
    st.write(
    """
    ###Heart rate and Altitude
    """
    )
    image = Image.open('./images/hr_session_detail.png')
    #st.image(image, caption="Heart and altitude",use_column_width=True)
    