import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.palettes import Magma256, Viridis256
import sqlite3
from streamlit import session_state
import astropy.units as u
from orbitdict import *

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
header = st.columns((1,3,1))
with header[1]:
    st.title('Known nearby planets comprising a target sample for reflected light imaging.')
    # 
    #     st.write(':new_moon: :waxing_crescent_moon: :first_quarter_moon: :moon: :waxing_gibbous_moon: :full_moon: :waning_gibbous_moon: :last_quarter_moon: :waning_crescent_moon: :new_moon:')
header2 = st.columns((1,3,1))
with header2[1]:
    st.write(':new_moon: :waxing_crescent_moon: :first_quarter_moon: :moon: :waxing_gibbous_moon: :full_moon: :waning_gibbous_moon: :last_quarter_moon: :waning_crescent_moon: :new_moon:  :waxing_crescent_moon: :first_quarter_moon: :moon: :waxing_gibbous_moon: :full_moon: :waning_gibbous_moon: :last_quarter_moon: :waning_crescent_moon: :new_moon:  :waxing_crescent_moon: :first_quarter_moon: :moon: :waxing_gibbous_moon: :full_moon: :waning_gibbous_moon: :last_quarter_moon: :waning_crescent_moon: :new_moon:  :waxing_crescent_moon: :first_quarter_moon: :moon: :waxing_gibbous_moon: :full_moon: :waning_gibbous_moon: :last_quarter_moon: :waning_crescent_moon: :new_moon:  :waxing_crescent_moon: :first_quarter_moon: :moon: :waxing_gibbous_moon: :full_moon: :waning_gibbous_moon: :last_quarter_moon: :waning_crescent_moon: :new_moon:  ')

st.markdown(
    """
    This plot shows 100s of the nearest ($<$70 pc) known RV-detected planets in the [Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) (as of Feb 2025), plotted as a function of separation, contrast, and phase for GMagAO-X on the GMT.  For planets without inclinitation values in the Archive, we used inclination = $60^{o}$, the average inclination for a uniform half-sphere.  If radius was not available in the Exoplanet Archive, we used a [Mass/Radius relation](https://github.com/logan-pearce/Reflected-Light/blob/main/Jareds-planet-mass-radius-daig.pdf); if mass was not available we used Msini.  Separation is in units of $\lambda/D$, the fundamental length scale for direct imaging (1 $\lambda/D$ ~ FWHM of PSF core). Contrast shown here is the reflected light only contrast for a uniform Lambertian sphere at the physical separation, phase, uniform albedo, and radius in the database; no albedo variation with wavelength or thermal emission is incorporated here.

    <---- Details of how this target list was compiled, assumptions made, and notebooks for generating this list yourself are in the "Derivations" tab.

    Hover over points to get information about each planet. You can zoom and pan using the buttons in the top right. You can adjust the wavelength, the primary diameter, and the geometric albedo using the sliders below. For example, the see how planets would appear for Roman CGI, slide the primary diameter tab to 2.4 meters; for ELT, slide it to 39 m.
    
    Planets highlighted in orange have orbit solutions in the "Predict Planet Location" tab.

    <---- The "Predict Planet Location" tab uses the [`projecc`](https://github.com/logan-pearce/projecc) package to turn literature orbit solutions into a prediction of planet location in the sky plane at a specified date.

    You can also select objects using the SQL interface which will automatically update the plot. 
    
    <---- Details and examples of SQL queries and planet orbits in the Read Me tab
"""
)

import sqlite3
import pandas as pd

aodb = pd.read_csv('Target-list-with-orbital-params.csv')


import sqlite3
from sqlalchemy import create_engine, text
conn = sqlite3.connect('targets.db')
engine = create_engine("sqlite:///targets.db")
aodb.to_sql(name = 'targets', con=engine, index=False, if_exists='replace')


def querySQL(string):
    with engine.connect() as conn:
        result = conn.execute(text(string)).fetchall()
        session_state['db'] = pd.DataFrame(result)
        st.dataframe(session_state['db'])


st.text_input(r"$\textsf{\Large SQL Query String}$", key='sqlquerystring')

session_state['db'] = aodb

#session_state
if session_state['sqlquerystring'] == '':
    session_state['db'] = aodb
    st.dataframe(session_state['db'])
else:
    with st.form(key="aodbsql"):
        st.form_submit_button('Query', on_click=querySQL(session_state['sqlquerystring']))


############################## Bokeh

def GetPointsWithinARegion(xdata, ydata, points):
    ''' For a region defined by points, return the indicies of items from [xdata,ydata]
    that lie within that region
    
    Args:
        xdata, ydata (arr): x and y data 
        points (arr): array of points describing region in tuples of (x,y)
        
    Returns:
        indicies of points in dataframe that lie within the region.
    '''

    # find points that lie within region:
    stacked1 = np.stack((xdata,ydata),axis=1)
    from matplotlib import path
    pp = path.Path(points)
    indicieswithinregion = pp.contains_points(stacked1)
    return indicieswithinregion


def MakeInteractiveSeparationContrastPlotOfNearbyRVPlanets(session_state, cont_curve = None):
    rad = session_state['db']['PlanetRadiuse'].copy()
    spt = session_state['db']['SpTNumber'].copy()
    plotx = session_state['db']['MaxProjectedSeparation_lod_gmagaox']
    ploty = session_state['db']['ContrastAtMaxProj']
    phases = session_state['db']['PhaseAtMaxProj']
    sepau = np.array(session_state['db']['MaxProjectedSeparation_au'])
    sepmas = np.array(session_state['db']['MaxProjectedSeparation_mas'])
    filt = 'i'
    xaxis_label = r'\[ \mathrm{Max\; Projected\; Separation}\; [\lambda/D]\]'
    annotation_text = ''
    IWA = 2
    ytop = 6e-6
    ybottom = 2e-10
    xright = 20
    xleft = 0
    ncolors = 10
    ticklocs = 'None'
    ticklabels = 'None'

    plotx, ploty = np.array(plotx),np.array(ploty)
    gmt_lod = 0.2063 * 0.8 / 24.5
    elt_lod = 0.2063 * 0.8 / 39
    mag_lod = 0.2063 * 0.8 / 6.5
    sep_elt = plotx*(gmt_lod/elt_lod)
    sep_mag = plotx*(gmt_lod/mag_lod)
    multiplier = 2
    datadf = pd.DataFrame(data={'plotx':plotx, 'ploty':ploty, 'color':spt, 'markersize':rad*multiplier,
                                'name':session_state['db']['pl_name'], 'rad':rad, 'spt':spt, 'dist':session_state['db']['sy_dist'],
                                'phases':phases, 'plotx_og':plotx, 'ploty_og':ploty, 'iwa': 2, 
                                'sepau':sepau, 'sepmas':sepmas, 'dec':session_state['db']['dec'], 
                                'starteff':session_state['db']['StarTeff'],
                                'masse':session_state['db']['pl_bmasse'],
                                'sep_elt':sep_elt, 'sep_mag':sep_mag, 'period':session_state['db']['pl_orbper'],
                                'stargaiamag':session_state['db']['sy_gaiamag']
                                })
    datadf = datadf.reset_index(drop=True)
    datadict = datadf.to_dict(orient = 'list')
    proxcendf = datadf.loc[np.where(datadf['name'] == 'Proxima Cen b')]
    proxcendict = proxcendf.to_dict(orient = 'list')

    from bokeh.plotting import figure, show, output_file, save
    from bokeh.io import output_notebook
    from bokeh.models import LinearColorMapper, ColumnDataSource, LinearInterpolator
    from bokeh.models import  Range1d, LabelSet, Label, ColorBar, FixedTicker, Span
    from bokeh.models import CustomJS, Slider
    from bokeh.layouts import column, row
    from bokeh.palettes import Magma256, Turbo256, brewer
    from bokeh.transform import linear_cmap
    #output_notebook()


    data=ColumnDataSource(data=datadict)
    proxima=ColumnDataSource(data=proxcendict)

    tools = "hover, box_zoom, zoom_in, zoom_out, save, undo, redo, reset, pan"
    tooltips = [
        ('Planet', '@name'),
        ('Cont', '@ploty'),
        ('Phase [deg]', '@phases{0}'),
        ('Period [days]', '@period{0.0}'),
        ("Sep [GMT i' lod]", '@plotx{0.0}'),
        ("Sep [ELT i' lod]", '@sep_elt{0.0}'),
        ("Sep [MagAO-X i' lod]", '@sep_mag{0.0}'),
        ('Sep [au]', '@sepau{0.00}'),
        ('Sep [mas]', '@sepmas{0.00}'),
        ('Rad [Rearth]','@rad{0.00}'),
        ('Mass or Msini [Mearth]','@masse{0.0}'),
        ('Star Teff', '@starteff{0}'),
        ('SpT','@spt{0.0}'),
        ('Star Gaia G', '@stargaiamag{0.0}'),
        ('Dist [pc]','@dist{0.0}'),
        ('Decl', '@dec{0.0}')
    ]

    p = figure(width=900, height=750, y_axis_type="log", tools=tools, 
                tooltips=tooltips, toolbar_location="above")

    mapper = linear_cmap(field_name='phases', 
                            palette=brewer['RdBu'][ncolors],
                            low=20, high=150)

    p.scatter('plotx','ploty', source=data, fill_alpha=0.8, size='markersize', 
                line_color=mapper, color=mapper)

    p.scatter('plotx','ploty', source=proxima, fill_alpha=0.8, size='markersize', 
                line_color='red', color=mapper)

    color_bar = ColorBar(color_mapper=mapper['transform'], width=15, 
                            location=(0,0), title="Phase",
                        title_text_font_size = '20pt',
                            major_label_text_font_size = '15pt')


    p.add_layout(color_bar, 'right')

    keys = [key for key in plDict.keys() if '---' not in key]
    names = np.array(data.data['name'])
    try:
        ind = []
        for key in keys:
            ind.append(int(np.where(names == key)[0][0]))
        datadfpoints = pd.DataFrame(data={'plotx':plotx[ind], 'ploty':ploty[ind], 'markersize':rad[ind]*multiplier,
                                            'phases':phases[ind], 'color':spt[ind], 
                                    'name':session_state['db']['pl_name'][ind], 'rad':rad[ind], 
                                    'spt':spt[ind], 'dist':session_state['db']['sy_dist'][ind],
                                        'phases':phases[ind], 'plotx_og':plotx[ind], 'ploty_og':ploty[ind], 'iwa': 2, 
                                        'sepau':sepau[ind], 'sepmas':sepmas[ind], 'dec':session_state['db']['dec'][ind], 
                                        'starteff':session_state['db']['StarTeff'][ind],
                                        'masse':session_state['db']['pl_bmasse'][ind],'period':session_state['db']['pl_orbper'][ind],
                                        'sep_elt':sep_elt[ind], 'sep_mag':sep_mag[ind],'stargaiamag':session_state['db']['sy_gaiamag'][ind]
                                    })
        datadfpoints = datadfpoints.reset_index(drop=True)
        datadfpointsdict = datadfpoints.to_dict(orient = 'list')
        datapoints=ColumnDataSource(data=datadfpointsdict)
        p.scatter('plotx','ploty', source=datapoints, fill_alpha=1, size='markersize', 
                    line_color='orangered', color=None, line_width=2)
    except IndexError:
        pass

    if cont_curve is None:
        pass
    else:
        #gmt_lod = (0.2063 * 0.8 / 24.5) * 1000
        #cont_curve[0] = [cont_curve[0][i]/gmt_lod for i in range(len(cont_curve[0]))]
        cont_curve[0].append(max(data.data['plotx']))
        cont_curve[0].append(max(data.data['plotx']))
        cont_curve[0] = [cont_curve[0][0]] + cont_curve[0]
        cont_curve[0] = cont_curve[0] + [cont_curve[0][0]]
        cont_curve[1].append(cont_curve[1][len(cont_curve[1])-1])
        cont_curve[1].append(1e-4)
        cont_curve[1] = [1e-4]+cont_curve[1]
        cont_curve[1] = cont_curve[1] + [1e-4]
        cont_curve = np.array(cont_curve).T
        
        
        p.line(np.array(cont_curve[:,0]),cont_curve[:,1])
    
        # points = GetPointsWithinARegion(data.data['plotx'], data.data['ploty'], cont_curve)
        # datadfpoints = pd.DataFrame(data={'plotx':plotx[points], 'ploty':ploty[points], 'markersize':rad[points]*multiplier,
        #                                   'phases':phases[points], 'color':spt[points], 
        #                            'name':session_state['db']['pl_name'][points], 'rad':rad[points], 
        #                            'spt':spt[points], 'dist':session_state['db']['sy_dist'][points],
        #                             'phases':phases[points], 'plotx_og':plotx[points], 'ploty_og':ploty[points], 'iwa': 2, 
        #                             'sepau':sepau[points], 'sepmas':sepmas[points], 'dec':session_state['db']['dec'][points], 
        #                             'starteff':session_state['db']['StarTeff'][points],
        #                             'masse':session_state['db']['pl_bmasse'][points],'period':session_state['db']['pl_orbper'][points],
        #                             'sep_elt':sep_elt[points], 'sep_mag':sep_mag[points],'stargaiamag':session_state['db']['sy_gaiamag'][points]
        #                            })
        # datadfpoints = datadfpoints.reset_index(drop=True)
        # datadfpointsdict = datadfpoints.to_dict(orient = 'list')
        # datapoints=ColumnDataSource(data=datadfpointsdict)
        # p.scatter('plotx','ploty', source=datapoints, fill_alpha=1, size='markersize', 
        #          line_color='black', color=None, line_width=2)


    p.xaxis.axis_label = xaxis_label
    p.yaxis.axis_label = r'\[ \mathrm{Planet/Star\; Reflected\; Light\; Flux\; Ratio} \]'
    p.xaxis.axis_label_text_font_size = '20pt'
    p.yaxis.axis_label_text_font_size = '20pt'
    p.yaxis.major_label_text_font_size = "15pt"
    p.xaxis.major_label_text_font_size = "15pt"

    iwa = Span(location=IWA,
                                dimension='height', line_color='grey',
                                line_dash='dashed', line_width=3)

    p.add_layout(iwa)

    p.x_range=Range1d(xleft,xright)
    p.y_range=Range1d(ybottom,ytop)


    AgSlider = Slider(start=0.05, end=1.0, value=0.3, step=.01, title="Geometric Albedo")
    IWASlider = Slider(start=1, end=10, value=2, step=.5, title="IWA")
    LambdaSlider = Slider(start=400, end=2000, value=800, step=50, title="Wavelength [nm]")
    DSlider = Slider(start=2, end=45, value=25.4, step=0.5, title="Primary Mirror Diameter [m]")

    sliders_callback_code = """
        var Ag = Ag.value;
        var Lambda = Lambda.value;
        var D = D.value;
        
        var lod = 6.3;
        var newlod = ((Lambda/1000) / D) * 1000
        
        var y = source.data['ploty_og'];
        var x = source.data['plotx_og'];
        var newy = y.map(m => m * Ag/0.3 );
        var newx = x.map(b => b * 800/Lambda );
        var newx = newx.map(d => d * D/25.4 );


        console.log(newy)
        console.log(newx)
        source.data['ploty'] = newy;
        source.data['plotx'] = newx;
        source.change.emit();
    """

    slider_args = dict(source=data, Ag=AgSlider, Lambda=LambdaSlider, D=DSlider)

    AgSlider.js_on_change('value', CustomJS(args=slider_args,code=sliders_callback_code))
    LambdaSlider.js_on_change('value', CustomJS(args=slider_args,code=sliders_callback_code))
    DSlider.js_on_change('value', CustomJS(args=slider_args,code=sliders_callback_code))

    slider_args2 = dict(source=proxima, Ag=AgSlider, Lambda=LambdaSlider, D=DSlider)

    AgSlider.js_on_change('value', CustomJS(args=slider_args2,code=sliders_callback_code))
    LambdaSlider.js_on_change('value', CustomJS(args=slider_args2,code=sliders_callback_code))
    DSlider.js_on_change('value', CustomJS(args=slider_args2,code=sliders_callback_code))
    try:
        slider_args3 = dict(source=datapoints, Ag=AgSlider, Lambda=LambdaSlider, D=DSlider)

        AgSlider.js_on_change('value', CustomJS(args=slider_args3,code=sliders_callback_code))
        LambdaSlider.js_on_change('value', CustomJS(args=slider_args3,code=sliders_callback_code))
        DSlider.js_on_change('value', CustomJS(args=slider_args3,code=sliders_callback_code))
    except UnboundLocalError:
        pass

    st.bokeh_chart(column(p, row(AgSlider),row(LambdaSlider),row(DSlider)), use_container_width=True)
    #st.bokeh_chart(p, use_container_width=True)

if 'cont_curve' not in session_state:
    session_state['cont_curve'] = None

######### Render the plot
if 'show_text' not in st.session_state:
    st.session_state.show_text = False
def toggle_image():
    st.session_state.show_text = not st.session_state.show_text

st.button(r"$\textsf{\Large Add a contrast curve}$", on_click=toggle_image)

if st.session_state.show_text:
    '''Enter contrast curve values to display on the above plot and select planets above the curve.  Enter a list of separation values in 
     milliarcseconds and a list of flux contrast values, separated by commas. '''
    row_input = st.columns((1,1))
    with row_input[0]:
        cont_curve_seps = st.text_input(r"$\textsf{\Large Separation [lambda/D]}$",key='cont_curve_seps')
    with row_input[1]:
        cont_curve_flux = st.text_input(r"$\textsf{\Large Flux contrast}$",key='cont_curve_flux')
    
    #st.write(cont_curve_seps, cont_curve_flux)


    if st.button(r"$\textsf{\Large Enter Contrast Curve}$", key='generate2'):
        cont_curve_seps = cont_curve_seps.split(',')
        cont_curve_seps = [float(cc.replace(' ','')) for cc in cont_curve_seps]
        cont_curve_flux = cont_curve_flux.split(',')
        cont_curve_flux = [float(cc.replace(' ','')) for cc in cont_curve_flux]
        cont_curve = [cont_curve_seps,cont_curve_flux]
        session_state['cont_curve'] = cont_curve
        MakeInteractiveSeparationContrastPlotOfNearbyRVPlanets(session_state, cont_curve = session_state['cont_curve'])
        #st.write(':sparkles: Added! :sparkles:')
    else:
       pass

if session_state['cont_curve'] == None:
    MakeInteractiveSeparationContrastPlotOfNearbyRVPlanets(session_state, cont_curve = session_state['cont_curve'])
else:
    pass

    