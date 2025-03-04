import streamlit as st
import pandas as pd


st.set_page_config(
        page_title="Reflected Light Planets",
        page_icon="images/star-only-orange-transp.png",
        layout="wide",
    )

sidebar_logo = 'images/star-only-orange-transp.png'
st.logo(sidebar_logo, size='large')

# left_co, cent_co,last_co = st.columns(3)
# with cent_co:
#     st.image('images/logo.png', width=300)


refDict = {
        'Faria+ 2022':"https://ui.adsabs.harvard.edu/abs/2022A%26A...658A.115F/abstract",
        'Suarez Mascareno+ 2020':'https://ui.adsabs.harvard.edu/abs/2020A%26A...639A..77S/abstract',
        'Luhn+ 2019':"https://ui.adsabs.harvard.edu/abs/2019AJ....157..149L/abstract",
        'Kane+ 2015':"https://ui.adsabs.harvard.edu/abs/2015ApJ...806...60K/abstract",
        'Butler+ 2006':"https://ui.adsabs.harvard.edu/abs/2006ApJ...646..505B/abstract",
        'Gaia DR3 NSS':"https://ui.adsabs.harvard.edu/abs/2023A%26A...674A..34G/abstract",
        'Rivera+ 2010':"https://ui.adsabs.harvard.edu/abs/2010ApJ...719..890R/abstract",
        'Benedict+ 2002':"https://ui.adsabs.harvard.edu/abs/2002ApJ...581L.115B/abstract",
        'Butler+ 2001':"https://ui.adsabs.harvard.edu/abs/2001ApJ...555..410B/abstract",
        'McArthur+ 2014':"https://ui.adsabs.harvard.edu/abs/2014ApJ...795...41M/abstract",
        'Wittenmyer+ 2009':"https://ui.adsabs.harvard.edu/abs/2009ApJS..182...97W/abstract",
        'Vogt+ 2005':"https://ui.adsabs.harvard.edu/abs/2005ApJ...632..638V/abstract",
        'Vogt+ 2010':"https://ui.adsabs.harvard.edu/abs/2010ApJ...708.1366V/abstract",
        "Cretignier+ 2023":"https://ui.adsabs.harvard.edu/abs/2023A%26A...678A...2C/abstract",
        "Howard+ 2011":"https://ui.adsabs.harvard.edu/abs/2011ApJ...730...10H/abstract",
        "Pepe+ 2011":"https://ui.adsabs.harvard.edu/abs/2011A%26A...534A..58P/abstract",
        'Trifonov+ 2018':"https://ui.adsabs.harvard.edu/abs/2018A%26A...609A.117T/abstract",
        'Feng+ 2017':"https://ui.adsabs.harvard.edu/abs/2017AJ....154..135F/abstract",
        'Burt+ 2021':"https://ui.adsabs.harvard.edu/abs/2021AJ....161...10B/abstract",
        'Kaminski+ 2018':"https://ui.adsabs.harvard.edu/abs/2018A%26A...618A.115K/abstract",
        'Thompson+ 2025':'https://arxiv.org/abs/2502.20561',
        'Gregory & Fischer 2010':'https://ui.adsabs.harvard.edu/abs/2010MNRAS.403..731G/abstract',
        'Wittenmyer+ 2007':'https://ui.adsabs.harvard.edu/abs/2007ApJ...654..625W/abstract',
        'Vogt+ 2015':"https://ui.adsabs.harvard.edu/abs/2015ApJ...814...12V/abstract",
        'Johnson+ 2016':"https://ui.adsabs.harvard.edu/abs/2016ApJ...821...74J/abstract"
    }
df = pd.DataFrame({'Reference':refDict.keys(),'url':[refDict[key] for key in refDict.keys()]}
)
df = df.sort_values('Reference')
df = df.reset_index(drop=True)

st.dataframe(df, column_config={"url": st.column_config.LinkColumn()})