import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from streamlit import session_state
import astropy.units as u
from projecc import *
import re


st.set_page_config(
        page_title="Reflected Light Planets",
        page_icon="images/star-only-orange-transp.png",
        layout="wide",
    )

sidebar_logo = 'images/star-only-orange-transp.png'
st.logo(sidebar_logo, size='large')

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image('images/black-transp-wordsonly.png', width=300)



st.write('## Use `projecc` to predict the location of a planet at a given time from literature solutions.')
'''

Written by Logan Pearce, 2025
https://github.com/logan-pearce/projecc; http://www.loganpearcescience.com

'''
gmt_lod = (0.2063 * 0.8 / 24.5) * 1000
elt_lod = (0.2063 * 0.8 / 39) * 1000
mag_lod = (0.2063 * 0.8 / 6.5) * 1000
cgi_lod = (0.2063 * 0.8 / 2.4) * 1000

def GetOrbitLocOnDate(planet, obstime):
        # find the most recent time of periastron to that date:
        ind = np.where(planet.periastron_times <= obstime)[0]
        nearest_periastron = planet.periastron_times[ind[-1]]
        # Using the mean orbit parameter values generate an array of separations spanning one orbit:
        Orbfrac = (obstime - nearest_periastron) / (planet.period[0]*u.d.to(u.yr))
        if type(planet.inc) == list:
            incl = planet.inc[0]
        elif type(planet.inc) == int:
            incl = planet.inc
        else:
            incl = 60
        if type(planet.lan) == list:
            lan = planet.lan[0]
        elif type(planet.lan) == int:
            lan = planet.lan
        else:
            lan = 0
        # Compute kepler's constant depending on if provided mass is Mpsini or Mp:
        try:
            kep = KeplersConstant(planet.Mstar[0]*u.Msun,(planet.Mpsini[0]/np.sin(np.radians(incl)))*u.Mearth)
        except AttributeError:
            kep = KeplersConstant(planet.Mstar[0]*u.Msun,planet.Mp[0]*u.Mearth)
        pos, vel, acc = KeplerianToCartesian(planet.sma[0],
                                                planet.ecc[0],
                                                incl,
                                                planet.argp[0],
                                                lan,
                                                [Orbfrac*2*np.pi],kep, solvefunc = DanbySolve)
        decs1 = ((pos[0].value / planet.distance[0])*1000)
        ras1 = ((pos[1].value / planet.distance[0])*1000)
        seps1 = ((np.sqrt(pos[0].value**2 + pos[1].value**2)/ planet.distance[0])*1000)
        
        return ras1, decs1, seps1

def MakePlot(planet, date, lim, plot_expected_position = True, plot_aperture = True, aperture_radius = gmt_lod):
    points = OrbitSim(planet, date)
    # set up some plotting parameters:
    import matplotlib as mpl
    mpl.rcParams['axes.titlesize'] = 20.0
    mpl.rcParams['axes.labelsize'] = 20.0
    mpl.rcParams['axes.linewidth'] = 1.2
    mpl.rcParams['font.family'] = 'serif'
    #mpl.rcParams['text.usetex'] = True
    mpl.rcParams['xtick.labelsize'] = 18.0
    mpl.rcParams['ytick.labelsize'] = 18.0

    fig = MakeCloudPlot(points, lim = lim, figsize = (9,7))
    ras1, decs1, seps1 = GetOrbitLocOnDate(planet, date)
    if plot_expected_position:
        fig.axes[0].scatter(ras1,decs1,
                    color='pink', lw=2)
    if plot_aperture:
        sep = np.sqrt(
                (points.ra_mas - ras1)**2 +
                (points.dec_mas - decs1)**2
                )
        from photutils.aperture import CircularAperture
        aper = CircularAperture([ras1,decs1],
                            aperture_radius)
        aper.plot(color='pink', lw=2)
        frac = np.where(sep < aperture_radius)[0].shape[0]/sep.shape[0]
        stdev = np.std(sep)
    else:
        frac = np.nan
        stdev = np.nan
    
    return fig, frac, stdev, seps1


if "date_max_elong" not in st.session_state:
    st.session_state.date_max_elong = True
    st.session_state.Mp_is_Mpsini = True
    st.session_state.plot_exp_pos = True
    st.session_state.plot_aperture = True
    st.session_state.aperture = gmt_lod

rows = st.columns((1,1))
with rows[0]:
    st.write('#### Enter orbital parameter mean and std deviation separated by commas in the unit specified:')
    with st.form("my_form"):
        
        collect_numbers = lambda x : [float(i) for i in re.split(",", x) if i != ""]
        

        rows2 = st.columns((1,1))
        with rows2[0]:
            sma = st.text_input("sma [au]:", key='sma')
            sma = collect_numbers(sma)

            ecc = st.text_input("ecc:", key='ecc')
            ecc = collect_numbers(ecc)

            Period = st.text_input("period [days]:", key='period')
            Period = collect_numbers(Period)

            argp = st.text_input("planet argp [deg]:", key='argp')
            argp = collect_numbers(argp)

            Mpsini = st.text_input("planet mass [M_earth]:", key='Mpsini')
            Mpsini = collect_numbers(Mpsini)

            plx = st.text_input("parallax [mas]:", key='plx')
            plx = collect_numbers(plx)

        with rows2[1]:
            t0 = st.text_input("T0 [JD]:", key='t0')
            t0 = collect_numbers(t0)

            incl = st.text_input("inc [deg]:", 'nan, nan', key='inc')
            if incl == 'nan, nan' or incl == 'nan':
                incl = np.nan
            else:
                incl = collect_numbers(incl)
                if len(incl) == 1:
                    incl = incl[0]
                else:
                    pass

            lan = st.text_input("lan [deg]:", '0', key='lan')
            if lan == 'nan':
                lan = np.nan
            else:
                lan = collect_numbers(lan)
                if len(lan) == 1:
                    lan = lan[0]
                else:
                    pass

            Mstar = st.text_input("Mstar [M_sun]:", key='Mstar')
            Mstar = collect_numbers(Mstar)

            st.radio(
                "Planet mass is Mp*sin(i)",
                key="Mp_is_Mpsini",
                options=[True, False], )

        rows2 = st.columns((1,1))
        with rows2[0]:
            obsdate = st.text_input("Obs Date [decimalyear or YYYY-MM-DD]:")
            if '-' in obsdate:
                    from astropy.time import Time
                    obsdate = Time(obsdate+'T00:00:00', format = 'isot')
                    obsdate = obsdate.decimalyear
            
        with rows2[1]:
            st.radio(
                "Use date of max elongation",
                key="date_max_elong",
                options=[True, False], )

        rows3 = st.columns((1,1))
        with rows3[0]:
            lim = st.number_input("Plot axes limit [mas]:", key='lim', step=1)
            st.radio(
                "Plot aperture",
                key="plot_aperture",
                options=[True, False], )
            
        with rows3[1]:
            st.radio(
                "Plot expected position",
                key="plot_exp_pos",
                options=[True, False], )
            
            aperture = st.selectbox(
                "Aperture size",
                ("GMT", "ELT", "Mag"),
                key="aperture_size"
                )
            if aperture == 'ELT':
                st.session_state.aperture = elt_lod
            elif aperture == 'Mag':
                st.session_state.aperture = mag_lod
            else:
                st.session_state.aperture = gmt_lod

        submitted = st.form_submit_button("Submit")
        if submitted:
            with rows[1]:
                pl = Planet(sma,ecc,incl,argp,lan,Period,t0,Mpsini,Mstar,plx,Mp_is_Mpsini = st.session_state.Mp_is_Mpsini)
                if st.session_state.date_max_elong:
                    date = pl.date_of_max_elongation
                else:
                    date = float(obsdate)

                if st.session_state.lim == 0:
                    lim = max(pl.seps_mean_params) + 0.3*max(pl.seps_mean_params)
                else:
                    lim = st.session_state.lim
                
                fig, frac = MakePlot(pl, date, lim, 
                    plot_expected_position = True, 
                    plot_aperture = st.session_state.plot_aperture, 
                    aperture_radius = st.session_state.aperture)
                st.pyplot(fig)
                st.write('Fraction of points within aperture: {:.2f}'.format(frac))
                """
                Mp/Mpsini [Mearth]: """+str(plDict[planetselect][solutionselect]['Mpsini'])+"""\n
                sma [au]: """+str(plDict[planetselect][solutionselect]['sma'])+"""\n
                ecc: """+str(plDict[planetselect][solutionselect]['ecc'])+"""\n
                inc [deg]: """+str(plDict[planetselect][solutionselect]['inc'])+"""\n
                argp [deg]: """+str(plDict[planetselect][solutionselect]['argp'])+"""\n
                lan [deg]: """+str(plDict[planetselect][solutionselect]['lan'])+"""\n
                period [days]: """+str(plDict[planetselect][solutionselect]['Period'])+"""\n
                T0 [JD]: """+str(plDict[planetselect][solutionselect]['t0'])+"""\n
                Mstar [Msun]: """+str(plDict[planetselect][solutionselect]['Mstar'])+"""\n
                plx [mas]: """+str(plDict[planetselect][solutionselect]['plx'])

with rows[1]:
    st.write('#### Or select a planet and solution from the drop down menu:')

    from orbitdict import *

    planetselect = st.selectbox(
        "Select planet",
        ([key for key in plDict.keys()]),index=None,
        placeholder="Select planet", label_visibility='collapsed', key='planetselect'
        )
    
    if planetselect != None:
        solutionselect = st.selectbox(
            "Select solution",
            ([key for key in plDict[planetselect].keys()]),index=None,
            placeholder="Select orbit solution", label_visibility='collapsed'
            )
    if planetselect != None:
        if solutionselect != None:
            if plDict[planetselect][solutionselect]['lan'] == 0:
                rows4 = st.columns((1,1))
                with rows4[0]:
                    st.write('Long of nodes is unconstrained. Select "nan" to allow all possible values, select "0" to limit lan to 0 deg only')
                with rows4[1]:
                    st.radio("Set lan",
                        key="set_lan",
                        options=[np.nan,0])

    if planetselect != None:
        if solutionselect != None:
            rows5 = st.columns((1,1))
            with rows5[1]:
                obsdate = st.text_input("Obs Date [decimalyear or YYYY-MM-DD]:")
                if '-' in obsdate:
                    from astropy.time import Time
                    obsdate = Time(obsdate+'T00:00:00', format = 'isot')
                    obsdate = obsdate.decimalyear
                
            with rows5[0]:
                st.radio(
                    "Use date of max elongation",
                    key="date_max_elong2",
                    options=[True, False], )

    if planetselect != None:
        if solutionselect != None:
            rows6 = st.columns((1,1))
            with rows6[0]:
                if plDict[planetselect][solutionselect]['lan'] == 0:
                    if np.isnan(st.session_state.set_lan):
                        st.radio(
                                "Plot aperture",
                                key="plot_aperture2",
                                options=[True, False], index=1)
                    else:
                            st.radio(
                                "Plot aperture",
                                key="plot_aperture2",
                                options=[True, False], )
                else:
                    st.radio(
                                "Plot aperture",
                                key="plot_aperture2",
                                options=[True, False], )
            with rows6[1]:
                aperture = st.selectbox(
                    "Aperture size",
                    ("GMT", "ELT", "Mag", "Roman"),
                    key="aperture_size2"
                    )
            if aperture == 'ELT':
                st.session_state.aperture2 = elt_lod
            elif aperture == 'Mag':
                st.session_state.aperture2 = mag_lod
            elif aperture == 'Roman':
                st.session_state.aperture2 = cgi_lod
            else:
                st.session_state.aperture2 = gmt_lod

    submitted2 = st.button("Generate Plot")
    if submitted2:
        
        if plDict[planetselect][solutionselect]['lan'] == 0:
            pl = Planet(
                plDict[planetselect][solutionselect]['sma'],
                plDict[planetselect][solutionselect]['ecc'],
                plDict[planetselect][solutionselect]['inc'],
                plDict[planetselect][solutionselect]['argp'],
                st.session_state.set_lan,
                plDict[planetselect][solutionselect]['Period'],
                plDict[planetselect][solutionselect]['t0'],
                plDict[planetselect][solutionselect]['Mpsini'],
                plDict[planetselect][solutionselect]['Mstar'],
                plDict[planetselect][solutionselect]['plx'],
                Mp_is_Mpsini = plDict[planetselect][solutionselect]['Mp_is_Mpsini'])
            if st.session_state.lim == 0:
                lim = max(pl.seps_mean_params) + 0.3*max(pl.seps_mean_params)
            else:
                lim = st.session_state.lim
            if st.session_state.date_max_elong2:
                date = pl.date_of_max_elongation
            else:
                date = float(obsdate)
            if np.isnan(st.session_state.set_lan):
                plot_expected_position = False
            else:
                plot_expected_position = True
            fig, frac, stdev, meansep = MakePlot(pl, date, lim, 
                    plot_expected_position = plot_expected_position, 
                    plot_aperture = st.session_state.plot_aperture2, 
                    aperture_radius = st.session_state.aperture2)
            st.pyplot(fig)
        else:
            pl = Planet(
                plDict[planetselect][solutionselect]['sma'],
                plDict[planetselect][solutionselect]['ecc'],
                plDict[planetselect][solutionselect]['inc'],
                plDict[planetselect][solutionselect]['argp'],
                plDict[planetselect][solutionselect]['lan'],
                plDict[planetselect][solutionselect]['Period'],
                plDict[planetselect][solutionselect]['t0'],
                plDict[planetselect][solutionselect]['Mpsini'],
                plDict[planetselect][solutionselect]['Mstar'],
                plDict[planetselect][solutionselect]['plx'],
                Mp_is_Mpsini = plDict[planetselect][solutionselect]['Mp_is_Mpsini'])
            plot_expected_position =  True
            if st.session_state.lim == 0:
                lim = max(pl.seps_mean_params) + 0.3*max(pl.seps_mean_params)
            else:
                lim = st.session_state.lim
            if st.session_state.date_max_elong2:
                date = pl.date_of_max_elongation
            else:
                date = float(obsdate)
            fig, frac, stdev, meansep = MakePlot(pl, date, lim, 
                    plot_expected_position = plot_expected_position, 
                    plot_aperture = st.session_state.plot_aperture2, 
                    aperture_radius = st.session_state.aperture2)
            st.pyplot(fig)
        
        

        st.write('Fraction of points within aperture: {:.2f}'.format(frac))
        st.write('Ratio of std dev of separation of points from expected to expected separation: {:.2f}'.format(stdev/meansep))

        """
        Mp/Mpsini [Mearth]: """+str(plDict[planetselect][solutionselect]['Mpsini'])+"""\n
        sma [au]: """+str(plDict[planetselect][solutionselect]['sma'])+"""\n
        ecc: """+str(plDict[planetselect][solutionselect]['ecc'])+"""\n
        inc [deg]: """+str(plDict[planetselect][solutionselect]['inc'])+"""\n
        argp [deg]: """+str(plDict[planetselect][solutionselect]['argp'])+"""\n
        lan [deg]: """+str(lan)+"""\n
        period [days]: """+str(plDict[planetselect][solutionselect]['Period'])+"""\n
        T0 [JD]: """+str(plDict[planetselect][solutionselect]['t0'])+"""\n
        Mstar [Msun]: """+str(plDict[planetselect][solutionselect]['Mstar'])+"""\n
        plx [mas]: """+str(plDict[planetselect][solutionselect]['plx'])+"""\n
        decl [deg]: """+str(plDict[planetselect][solutionselect]['dec'])

        import io
        img = io.BytesIO()
        fig.savefig(img, format='png', dpi=300, bbox_inches='tight')
        btn = st.download_button(
            label="Download Figure",
            data=img,
            file_name='figure.png',
            mime="image/png"
    )



