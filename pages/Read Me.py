import streamlit as st
import pandas as pd

st.set_page_config(
        page_title="Ground-based Reflected Light Imaging Planner",
        page_icon="images/grip-offwhite.png",
        layout="wide",
    )

sidebar_logo = 'images/grip-offwhite.png'
st.logo(sidebar_logo, size='large')

st.sidebar.write("""<div style="width:100%;text-align:center;"><a href="https://docs.google.com/spreadsheets/d/1sk9wDIVi4uWELL_gSyr7Zb41HZJ27Puyvc1QqWBZFs4/edit?gid=0#gid=0" style="text-decoration:none; color:#EEEEEE">Indirectly Discovered, \n Directly Detected Companions</a></div>""", unsafe_allow_html=True)

header = st.columns((1,3,1))
with header[1]:
    st.title('Reflected Light Planets with projecc')


rows1 = st.columns((1,1))
with rows1[0]:
    """
    #### Citation
    If you use `projecc` or this app please cite Pearce, Males, and Limbach (2025, submitted to PASP)."""
    st.html('<a href="https://doi.org/10.5281/zenodo.15829821"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15829821.svg" alt="DOI"></a>') 
    """



    #### Contibuting
    Contributions welcome! To contribute please email Logan Pearce at lapearce@umich.edu"""

with rows1[1]:
    '''
    #### Future Upgrades
    - Permit asymmetric uncertainties and correlation matricies; interface with orbit fitter outputs like orbitize!
    - Overlay dark hole regions and compute fraction of points within dark hole
    - Compounding errors over time
    '''

"""
# Using the SQL database and the interactive plot

The database displayed on the front page is a list of RV-detected planets pulled from the Exoplanet Archive and filtered to provide the best set of planets for drawing a target list for reflected light imaging campaigns with ELTs and space-based platforms. How this target list was developed, and a notebook for reprpducing the list, is available in the 'Derivations' tab. 

This database is queryable with SQL. SQL is a language for selecting specific elements from a database. You can use the SQL interface above the database to query and filter the target list. After entering a query in the text box, the  app will display the results and everything on the main page will update to use only the results from the query. To return to the whole database, reload the page.

You can download the database or a query result using the tools in the upper right.

"""

if 'show_text' not in st.session_state:
    st.session_state.show_text = False
def toggle_example():
    st.session_state.show_text = not st.session_state.show_text

st.button('Show/Hide SQL Examples and db columns', on_click=toggle_example)

if st.session_state.show_text:
    rows = st.columns((2,1))
    with rows[0]:
        '''### Example basic SQL queries'''

        ''' #### Select every column for a specific planet '''
        strg = "SELECT * FROM targets WHERE pl_name = 'Prox Cen b'"
        st.code(strg, language="sql")
        '''Note: strings must be enclosed in quotes, floats or integrers should not'''

        ''' #### Select specific columns'''
        strg = "SELECT plx, ContrastAtMaxProj, PlanetRadiuse FROM targets WHERE pl_name = 'Prox Cen b'"
        st.code(strg, language="sql")
        '''Note: this will cause errors for the plots on the main page if you select only columns not displayed on the plot. This is fine.'''

        ''' #### Select planets around stars earlier than M '''
        strg = "SELECT * FROM targets WHERE SpTNumber < 6"
        st.code(strg, language="sql")

        ''' #### Select planets with radius less than 5 Rearth accessible from the GMT site: '''
        strg = "SELECT * FROM targets WHERE PlanetRadiuse < 5 AND dec < 20 AND dec > -65"
        st.code(strg, language="sql")

        ''' #### Select all the tau Ceti planets: '''
        strg = 'SELECT * FROM targets WHERE pl_name LIKE "%tau Cet%"'
        st.code(strg, language="sql")

    with rows[1]:
        '''Database Columns'''
        db = pd.read_csv('Target-list-with-orbital-params.csv')
        cols = pd.DataFrame(db.columns)
        st.dataframe(cols,width=400, height=600, hide_index=True)

'''
The plot below the database shows the targets as a function of planet-star maximum separation in lambda/D vs uniform Lamertian planet-star flux contrast, sized by their estimated radius. By default it shows this for lambda = 0.8 um, diameter = 24.5 m (GMT sized primary), and geometric albedo = 0.3. Sliders below the plot allow you to adjust these parameters. For example, to see the planets as they would appear for a 6.5 m telescope (such as Magellan Clay or JWST), slide the diameter slider to 6.5. The plot will automatically update to the results of the SQL query, so planets you select in the query will be displayed in the plot. The plot is interactive based on Bokeh, so you can zoom and pan using the tools in the upper right corner, and hover over points to see details of each planet.

You can also supply an estimated contrast curve to the plot. Click the "Add Contrast Curve" button, enter a list of separations in mas and corresponding contrast limits in flux units and hit enter.  The plot will update to show the curve.

'''
if 'show_contxample' not in st.session_state:
    st.session_state.show_contxample = False
def toggle_image():
    st.session_state.show_contxample = not st.session_state.show_contxample

st.button('Show/Hide Example of Contrast Curve', on_click=toggle_image)
if st.session_state.show_contxample:
    st.image('images/contcurve-example.png')

'''
# Using projecc to predict planet locations

The "Predict Planet Location" tab has a user interface for working with the `projecc` package to generate predition of a planet's location as a function of time from literature orbital parameter distrubutions.  

This diagram illustrates how `projecc` works. Users supply the mean and std of a Gaussian distribution for 8 parameters: semi-major axis ($a$) in au, eccentricity ($e$), period ($P$) in days, planet's argument of periastron ($\omega$) in degrees, host star mass ($M_*$) in Msun, planet mass or M*sin(i) in Earth masses, parallax (varpi) in mas, and epoch of periastron passage ($T_0$) in julian days.  Two parameters take flexible inputs.  For RV-only orbital solutions, inclination ($i$) and longitude of nodes ($\Omega$) are unknown; supplying NaN for either causes $i$ to be drawn from a cos(i) uniform distribution on [-1,1] and $\Omega$ to be drawn from uniform on [0,360] deg; supplying a tuple for either causes them to be drawn from a Gaussian; or a single value for either can be supplied. Users then supply a date for the simulated observation either in decimal degrees or YYYY-MM-DD.  `projecc` then simulates $N$ realizations of the planet's location in the sky plane drawn from the supplied parameter distrubutions.
'''
img = st.columns((1,3,1))
with img[1]:
    st.image('images/flowchart.png', width=700)




'''
There are two ways to use the app interface.

#### Manual parameter entry

Supply Gaussian mean, std values separated by a comma for each parameter in the units indicated to the form on the left side of the interface. By default $i$ is set to nan and $\Omega$ to zero. Indicate if the supplied planet mass represents a true mass or an M*sin(i) value. Supply a desired observation date, or alternately you can use the date at which the planet is expected to be at maximum separation from the star. If you elect to plot an aperture, `projecc` will compute the fraction of simulated realizations that fall within and aperture of radius 1$\lambda/D$ from the expected position and display this value below the returned figure. Once submitted, `projecc` will compute 100,000 simulated realizations of the planet's position on that date and display a plot on the right hand side, color coded by viewing phase. The fraction of points within an aperture and the spread of the points, as well as the entered parameter values, are displayed below the plot, along with a button for saving this figure if desired.'''

if 'show_example2' not in st.session_state:
    st.session_state.show_example2= False
def toggle_image2():
    st.session_state.show_example2 = not st.session_state.show_example2

st.button('Show/Hide Diagram of Manual Entry', on_click=toggle_image2)
if st.session_state.show_example2:
    img2 = st.columns((1,3,1))
    with img2[1]:
        st.image('images/example2.png', width=700)

'''
#### Supplied planets

Alternatively, you can select pre-loaded literature solutions for a set of planets. Planets in the main page plot that are highlighted in orange have literature solutions available to plot. Select a planet from the dropdown menu, then select a published set of orbital elements. If the published orbit solution doesn't constrain longitude of nodes, a radio button will appear to select if $\Omega$ will be drawn from U[0,360] or to fix it to zero. Then you can select to plot an aperture and the observation date.

'''

if 'show_example1' not in st.session_state:
    st.session_state.show_example1= False
def toggle_image1():
    st.session_state.show_example1 = not st.session_state.show_example1

st.button('Show/Hide Diagram of Supplied Planet Entry', on_click=toggle_image1)
if st.session_state.show_example1:
    img3 = st.columns((1,3,1))
    with img3[1]:
        st.image('images/example1.png', width=700)

'''
 Once submitted, `projecc` will compute 100,000 simulated realizations of the planet's position on that date and display a plot on the right hand side, color coded by viewing phase. The fraction of points within an aperture and the spread of the points, as well as the orbital parameter values, are displayed below the plot, along with a button for saving this figure if desired.

 We provide two summary statistics to quantify the confidence in locating a planet with a given literature solution. If selected, `projecc` will place an aperture at the expected location on the date specified, and display the fraction of simulated points within the aperture; a higher fraction is a more tightly constrained prediction. We also quantify the standard deviation of separations of simulated points and normalize the standard deviation by the expected separation; a lower number indicates points are more closely constrained to the expected separation. These statistics will be displayed only if "Plot aperture" is selected.

'''