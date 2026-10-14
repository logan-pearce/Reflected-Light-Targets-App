import numpy as np
import astropy.units as u


plDict = {
    '--- Gas Giants ---':{},
    'GJ 876 b':{
        'Rivera+ 2010':{'sma':[0.2083,0.00002], 'Period':[61.1166,0.0086], 'ecc':[0.0324,0.0013], 'Mpsini':[723.22,1.43],
                    'argp':[230.3,3.2], 'lan':0, 'inc':[61,2.4], 't0':[2450546.83, 0.54], 
                    'Mstar':[0.3696,0.0093], 'plx':[213.867,0.076],'dec':-14.266,
                    'Mp_is_Mpsini':False}, 
        'Benedict+ 2002':{'sma':[0.22,0.08], 'Period':[61.02,0.03], 'ecc':[0.1,0.02], 'Mpsini':[600,108],
                    'argp':[158.96,0.36], 'lan':[25,4], 'inc':[84,6], 't0':[2450107.9, 1.9], 
                    'Mstar':[0.3696,0.0093], 'plx':[213.867,0.076],'dec':-14.266,
                    'Mp_is_Mpsini':False},
        'Gaia DR3 NSS':{'sma':[0.3387952889421918, 0.0436074707046778], 'Period':[61.36053201535001, 0.22405064], 
                    'ecc':[0.1575966196274674, 0.15398227], 'Mpsini':[723.2235, 1.4302],
                    'argp':[78.27074632191767, 57.312635307790735], 'lan':[184.23836392262012, 4.5404789153926135], 
                    'inc':[101.46496611370219, 7.540971630586838], 't0':[2457381.3153060274, 19.39020347595215], 
                    'Mstar':[0.369607209603, 0.00925103268102], 'plx':[213.867, 0.076],'dec':-14.266,
                    'Mp_is_Mpsini':False},
                    },
    'HR 810 b':{
        'Butler+ 2001':{'sma':[0.91,1e-8], 'Period':[312,5], 'ecc':[0.15,0.05], 'Mpsini':[677,1e-8],
                    'argp':[129,5], 'lan':0, 'inc':[87,6], 't0':[2451493, 4], 
                    'Mstar':[1.34,0.02], 'plx':[57.7143000, 0.0536787],'dec':-50.799317,
                    'Mp_is_Mpsini':True},
                    },
    '70 Vir b':{
        'Luhn+ 2019':{'sma':[0.481,0.022], 'Period':[116.6880,0.0054], 'ecc':[0.3988,0.0019], 'Mpsini':[2357,18],
                    'argp':[179.83,0.29], 'lan':0, 'inc':np.nan, 't0':[2455291.721, 0.086], 
                    'Mstar':[1.09,0.00005], 'plx':[55.83,0.173],'dec':13.776,
                    'Mp_is_Mpsini':True},
        'Kane+ 2015':{'sma':[0.481,0.003], 'Period':[116.6926,0.0014], 'ecc':[0.399,0.002], 'Mpsini':[2352,6],
                    'argp':[178.8,0.3], 'lan':0, 'inc':np.nan, 't0':[2447239.70911, 0.11], 
                    'Mstar':[1.09,0.00005], 'plx':[55.83,0.173],'dec':13.776,
                    'Mp_is_Mpsini':True},
        'Butler+ 2006':{'sma':[0.484,0.028], 'Period':[116.6884,0.004], 'ecc':[0.4007,0.0035], 'Mpsini':[2380.45,193.87],
                    'argp':[178.71,0.54], 'lan':0, 'inc':np.nan, 't0':[2447239.82, 0.21], 
                    'Mstar':[1.09,0.00005], 'plx':[55.83,0.173],'dec':13.776,
                    'Mp_is_Mpsini':True},
                    },
    'HD 128311 b':{
        'McArthur+ 2014':{'sma':[1.084,0.006], 'Period':[453.019,0.404], 'ecc':[0.303,0.011], 'Mpsini':[562.2,7.3],
                    'argp':[237.8,3.3], 'lan':0, 'inc':np.nan, 't0':[2450198.691, 4.472], 
                    'Mstar':[0.81,0.09], 'plx':[61.21,0.07],'dec':9.745,
                    'Mp_is_Mpsini':True},
        'Wittenmyer+ 2009':{'sma':[1.086,0.008], 'Period':[454.2,1.6], 'ecc':[0.345,0.049], 'Mpsini':[460.83,41.32],
                    'argp':[243,16], 'lan':0, 'inc':np.nan, 't0':[2453835, 11], 
                    'Mstar':[0.81,0.09], 'plx':[61.21,0.07],'dec':9.745,
                    'Mp_is_Mpsini':True},
        'Vogt+ 2005':{'sma':[1.099,0.04], 'Period':[458.6,6.8], 'ecc':[0.25,0.1], 'Mpsini':[692.84,69.92],
                    'argp':[291,36], 'lan':0, 'inc':np.nan, 't0':[2450211, 76], 
                    'Mstar':[0.81,0.09], 'plx':[61.21,0.07],'dec':9.745,
                    'Mp_is_Mpsini':True},
                },
    'HD 114783 b':{
        'Wittenmyer+ 2009':{'sma':[1.16,0.019], 'Period':[493.7,1.8], 'ecc':[0.144,0.032], 'Mpsini':[349.6,19.07],
                    'argp':[266,11], 'lan':0, 'inc':np.nan, 't0':[2453806, 14], 
                    'Mstar':[0.85,0.1], 'plx':[47.44,0.06],'dec':-2.265,
                    'Mp_is_Mpsini':True},
        'Butler+ 2006':{'sma':[1.169,0.068], 'Period':[496.9,2.3], 'ecc':[0.085,0.033], 'Mpsini':[328.6,28.3],
                    'argp':[273,25], 'lan':0, 'inc':np.nan, 't0':[2450840, 37], 
                    'Mstar':[0.85,0.1], 'plx':[47.44,0.06],'dec':-2.265,
                    'Mp_is_Mpsini':True},
                 },
    'bet Pic b':{
        'Sutlieff+ 2026':{'sma':[10.07,0.03], 'Period':[23.77*u.yr.to(u.d),0.16*u.yr.to(u.d)], 'ecc':[0.105,0.004], 
                    'Mpsini':[8.7*u.Mjup.to(u.Mearth),2.6*u.Mjup.to(u.Mearth)],
                    'argp':[23.7,3.1], 'lan':[211.787,0.01], 'inc':[88.993,0.010], 't0':[2447991.05, 14.61], 
                    'Mstar':[1.789,0.027], 'plx':[50.6,0.3],'dec':-51.066,
                    'Mp_is_Mpsini':False, 'Rp':[18.5,0.7],#Currie+2013
                    'StarTeff':[7890,17], 'Rstar':[2.01,0.17], 'age':[0.0185, 0.003] #Teff:Swastik+2021, Rstar:GaiaDR2, Age:Kammerer+2024
                    },
                },
    'bet Pic c':{
        'Sutlieff+ 2026':{'sma':[2.7,0.03], 'Period':[1202,26], 'ecc':[0.307,0.025], 'Mpsini':[2765,250],
                    'argp':[-114.9 % 360, 2.6], 'lan':[211.08,0.05], 'inc':[88.93, 0.11], 't0':[2453806, 14], 
                    'Mstar':[1.789,0.027], 'plx':[50.6,0.3],'dec':-51.066,
                    'Mp_is_Mpsini':False, 'StarTeff':[7890,17], 'Rstar':[2.01,0.17], #Teff:Swastik+2021, Rstar:GaiaDR2
                    'age':[0.0185, 0.003]
                    },
        'Nowak+ 2020':{'sma':[2.72,0.02], 'Period':[1230.8925, 14.61], 'ecc':[0.37,0.12], 'Mpsini':[2606, 254],
                    'argp':[66.2,2.5], 'lan':[30.98,0.12], 'inc':[89.17,0.5], 't0':[2459911, 28], 
                    'Mstar':[1.82,0.03], 'plx':[50.62,0.33],'dec':-51,
                    'Mp_is_Mpsini':False},
        'Feng+ 2022':{'sma':[2.68,0.016], 'Period':[1192.8,5.5], 'ecc':[0.314,0.034], 'Mpsini':[3222.5,373],
                    'argp':[64, 3], 'lan':[31.081,0.06], 'inc':[88.947,0.09], 't0':[2452696.565,28], 
                    'Mstar':[1.82,0.03], 'plx':[50.62,0.33],'dec':-51,
                    'Mp_is_Mpsini':False}
                },
    '14 Her b':{
        'Bardalez Gagliuffi+ 2025':{'sma':[2.839,0.039], 'Period':[1765.0389,1.8], 'ecc':[0.3683,0.0029], 'Mpsini':[2835,540],
                    'argp':[202.68, 0.46], 'lan':[44.69,4], 'inc':[32.66,7.4], 't0':[2458853.207,0.006], 
                    'Mstar':[0.97,0.04], 'plx':[55.8657,0.0291],'dec':43.816,
                    'Mp_is_Mpsini':False, 'StarTeff':[5338,25], 'Rstar':[0.93,0.01],
                    'age':[4.6,3.8,1.3]} #Teff/Rstar from Stassun+2017
                },
    'HD 190360 b':{
            'Giovinazzi+ 2026':{'sma':[3.965,0.047], 'Period':[2887.5,3.6], 'ecc':[0.3342,0.0062], 'Mpsini':[489,79],
                        'argp':[202.68, 0.46], 'lan':[108,60], 'inc':[71,42], 't0':[2456423.6,8.9], 
                        'Mstar':[0.996,0.036], 'plx':[62.44,0.6],
                        'Mp_is_Mpsini':False, 'StarTeff':[5560,100], 'Rstar':[1.201,0.0327],
                        'age':[8.85, 6.57, 4.05]} #Teff: Giovvinazzi+2026, Rstar: Giovinazzi+2-25
                    },
    'HD 154345 b':{
                'Giovinazzi+ 2025':{'sma':[4.158,0.066], 'Period':[3280.31,28.85], 'ecc':[0.058,0.019], 'Mpsini':[376.94,30.19],
                            'argp':[309,21], 'lan':[68,31], 'inc':[88,20], 't0':[2458230,260], 
                            'Mstar':[0.89,0.04], 'plx':[54.7358942,0.000017],
                            'Mp_is_Mpsini':False, 'StarTeff':[5468,44], 'Rstar':[0.85,0.01],
                            'age':[3.1,0.4]} #Teff:Xiao+2023, Rstar:Stassun+2017
                        },
    'GJ 832 b':{
                'Philipot+ 2023':{'sma':[3.7,0.1], 'Period':[3853,51], 'ecc':[0.05,0.03], 'Mpsini':[0.99*u.Mjup.to(u.Mearth),0.09*u.Mjup.to(u.Mearth)],
                                'argp':[207,31], 'lan':[61,17], 'inc':[51,3], 't0':[2456696, 450], 
                                'Mstar':[0.45,0.01], 'plx':[201.3252,0.0237],
                                'Mp_is_Mpsini':False, 'StarTeff':[3993.4,120], 'Rstar':[0.44,0.013],
                                'age':[5.27,2.14]} #Teff:GaiaDR2, Rstar:TICv8
                            },
    'HD 114783 c':{
            'Philipot+ 2023':{'sma':[5.0,0.1], 'Period':[1618,88], 'ecc':[0.05,0.04], 'Mpsini':[1.9*u.Mjup.to(u.Mearth),0.5*u.Mjup.to(u.Mearth)],
                                            'argp':[153,48], 'lan':[307,14], 'inc':[21,7], 't0':[2455618,580], 
                                            'Mstar':[0.9,0.04], 'plx':[47.45,0.06],
                                            'Mp_is_Mpsini':False, 'StarTeff':[5100,67], 'Rstar':[0.833,0.02],
                                            'age':[6.2,4.9]}, #Teff/Rstar from GaiaDR2
                     },
    'HD 87883 b':{
            'Feng+ 2022':{'sma':[4.055,0.16], 'Period':[3309.22,30.46], 'ecc':[0.707,0.007], 'Mpsini':[5.409*u.Mjup.to(u.Mearth),0.63*u.Mjup.to(u.Mearth)],
                                            'argp':[289.351,1.289], 'lan':[99.951,4.2], 'inc':[25.4,1.6], 't0':[2450345.193,62], 
                                            'Mstar':[0.81,0.91], 'plx':[54.64,0.036],
                                            'Mp_is_Mpsini':False, 'StarTeff':[4915,73], 'Rstar':[0.79,0.03],
                                            'age':[9.1,3.5]}, #Teff/Rstar from Stassun+2017
                     },
    # 'HD 181433 d':{
    #         'Feng+ 2022':{'sma':[], 'Period':[], 'ecc':[], 'Mpsini':[*u.Mjup.to(u.Mearth),*u.Mjup.to(u.Mearth)],
    #                                         'argp':[], 'lan':[], 'inc':[], 't0':[], 
    #                                         'Mstar':[], 'plx':[5],
    #                                         'Mp_is_Mpsini':False},
    #                  },
    'pi Men b':{
                'Piccininini+ 2026':{'sma':[3.24,0.03], 'Period':[2089.0,0.4 ], 'ecc':[0.637,0.002], 'Mpsini':[12.4*u.Mjup.to(u.Mearth),0.3*u.Mjup.to(u.Mearth)],
                                                'argp':[330.1,0.03], 'lan':[282.5,4.3], 'inc':[51.1,3], 't0':[2458389,3], 
                                                'Mstar':[1.02,0.03], 'plx':[54.705200,0.067131 ],
                                                'Mp_is_Mpsini':False, 'StarTeff':[5870,50], 'Rstar':[1.1,0.01],
                                                'age':[5.10,4.75,1.51]}, #Teff/Rstar from Piccinini+2026
                         },
    'HD 75302 b':{
                    'Philipot+ 2023':{'sma':[5.3,0.2], 'Period':[4356,173], 'ecc':[0.39,0.11], 'Mpsini':[5.4*u.Mjup.to(u.Mearth),0.5*u.Mjup.to(u.Mearth)],
                                                    'argp':[307,37], 'lan':[211,11], 'inc':[29,5], 't0':[2457607,245], 
                                                    'Mstar':[1.06,0.05], 'plx':[32.9432,0.0652],
                                                    'Mp_is_Mpsini':False, 'StarTeff':[5697,130.], 'Rstar':[0.94,0.05]}, #Teff/Rstar from TICv8
                             },
    'HD 70642 b':{
                    'Xiao+ 2023':{'sma':[3.295,0.021], 'Period':[5.751*u.yr.to(u.d),0.038*u.yr.to(u.d)], 
                                  'ecc':[0.04,0.034], 'Mpsini':[3.9*u.Mjup.to(u.Mearth),0.29*u.Mjup.to(u.Mearth)],
                                                    'argp':[258,66], 'lan':[60.7,8.8], 'inc':[29.9,2.6], 't0':[2455918,358], 
                                                    'Mstar':[1.078,0.015], 'plx':[34.1287,0.0319],
                                                    'Mp_is_Mpsini':False, 'StarTeff':[5732,23], 'Rstar':[1.05,0.02],
                                                    'age':[0.92,0.73]}, #Teff:Xiao+2023, Rstar: GaiaDR2
                             },
    'HD 216437 b':{
                    'Philipot+ 2023':{'sma':[2.57,0.04], 'Period':[1359,1], 
                                  'ecc':[0.31,0.001], 'Mpsini':[4.2*u.Mjup.to(u.Mearth),0.7*u.Mjup.to(u.Mearth)],
                                                    'argp':[65,2], 'lan':[129,8], 'inc':[36,8], 't0':[2454667,8], 
                                                    'Mstar':[1.22,0.06], 'plx':[37.438,0.0447], 'StarTeff':[5905,31], 'Rstar':[1.48,0.1], 'age':[4.75,1.06],
                                                    'Mp_is_Mpsini':False},
                             },
    'gam Cep b':{
                'Endl+ 2011':{'sma':[2.05,0.06], 'Period':[903.3,1.5], 'ecc':[0.049,0.034], 
                                                  'Mpsini':[1.85*u.Mjup.to(u.Mearth),0.16*u.Mjup.to(u.Mearth)],
                                                                'argp':[(94.6+180)%360,34.6], 'lan':np.nan, 'inc':np.nan, 't0':[2453227,87], 
                                                                'Mstar':[1.4,0.12], 'plx':[70.91,0.4],
                                                                'Mp_is_Mpsini':True, 'StarTeff':[4775,31], 'Rstar':[4.74,0.08],
                                                                'age':[6.6,2.6,0.7]}, #Teff/Rstar from Knudstrup+2023
                'Benedict+ 2018':{'sma':np.nan, 'Period':905.0, 'ecc':0.08, 
                                  'Mpsini':[9.4*u.Mjup.to(u.Mearth),1.1*u.Mjup.to(u.Mearth)],
                                                'argp':86, 'lan':[47,6], 'inc':[169.5], 't0':2453248.1, 
                                                'Mstar':1.18, 'plx':[70.91,0.4],
                                                'Mp_is_Mpsini':False},
                         },
    'ups And d':{
                'Harada+ 2025':{'sma':[2.585,0.05], 'Period':[1281.4,0.73], 'ecc':[0.3139,0.0096], 
                                  'Mpsini':[1378.0,55],
                                                'argp':[72,1.8], 'lan':np.nan, 'inc':np.nan, 't0':[2455905.16, 7.97], 
                                                'Mstar':[1.29,0.04], 'plx':[74.57,0.35],
                                                'Mp_is_Mpsini':True, 'StarTeff':[6105.51,150], 'Rstar':[1.63,0.1],
                                                'age':[3.80,1]}, #Teff/Rstar from TICv8
                         },
    'nu Oph c':{
                'Teng+ 2023':{'sma':[5.829,0.006], 'Period':[3180,5.01], 'ecc':[0.179,0.007], 
                                  'Mpsini':[23.693*u.Mjup.to(u.Mearth),0.205*u.Mjup.to(u.Mearth)],
                                                'argp':[188.54,1.94], 'lan':np.nan, 'inc':np.nan, 't0':[2449885.9,17.6], 
                                                'Mstar':[2.61,0.57], 'plx':[21.64,0.26],
                                                'Mp_is_Mpsini':True},
                         },
    'HD 160691 c':{
                'Harada+ 2025':{'sma':[5.52,0.17], 'Period':[4241.0,160], 'ecc':[0.039,0.011], 
                                  'Mpsini':[2.262*u.Mjup.to(u.Mearth),0.2*u.Mjup.to(u.Mearth)],
                                                'argp':[292,22], 'lan':np.nan, 'inc':np.nan, 't0':[2452070,1380], 
                                                'Mstar':[1.13,0.02], 'plx':[64.08,0.12],
                                                'Mp_is_Mpsini':True, 'StarTeff':[5771.65,119], 'Rstar':[1.37,0.06], 
                                                'age':[5.666,4.401,6.963]},#Teff/Rstar from TICv8
                         },
    'HD 160691 b':{
                'Harada+ 2025':{'sma':[1.57,0.03], 'Period':[644.93,0.27], 'ecc':[0.0505,0.0094], 
                                  'Mpsini':[556.0,21],'argp':[(23+180)%360, 9.7], 'lan':np.nan, 'inc':np.nan,
                                't0':[2456995.449556262, 28.229556806442144], 
                                                'Mstar':[1.13,0.02], 'plx':[64.08,0.12],
                                                'Mp_is_Mpsini':True, 'StarTeff':[5771.65,119], 'Rstar':[1.37,0.06],
                                                'age':[5.666,4.401,6.963]},#Teff/Rstar from TICv8
                               },
    '47 UMa b':{
                'Harada+ 2025':{'sma':[2.104,0.04], 'Period':[1076.06,1.50], 'ecc':[0.03,0.008], 
                                  'Mpsini':[2.492*u.Mjup.to(u.Mearth),0.107*u.Mjup.to(u.Mearth)],
                                                'argp':[142,16], 'lan':np.nan, 'inc':np.nan, 't0':[2452408.5,4.2], 
                                                'Mstar':[1.063,0.029], 'plx':[72.45,0.15],
                                                'Mp_is_Mpsini':True, 'StarTeff':[5872,113], 'Rstar':[1.21,0.05],
                                                'age':[7.30,1.90]},#Teff/Rstar from TICv8
                         },
    '47 UMa c':{
                'Harada+ 2025':{'sma':[3.453,0.067], 'Period':[6.20*u.yr.to(u.d),0.04*u.yr.to(u.d)], 'ecc':[0.247,0.15], 
                                  'Mpsini':[0.516*u.Mjup.to(u.Mearth),0.0378*u.Mjup.to(u.Mearth)],
                                                'argp':[236,22], 'lan':np.nan, 'inc':np.nan, 't0':[2456240,470], 
                                                'Mstar':[1.063,0.029], 'plx':[72.45,0.15],
                                                'Mp_is_Mpsini':True, 'StarTeff':[5872,113], 'Rstar':[1.21,0.05], 'age':[7.30,1.90]},
                'Gregory & Fischer 2010':{'sma':[3.6,0.1], 'Period':[2391,100], 'ecc':[0.098,0.1], 'Mpsini':[171.621,23],
                    'argp':[115, 160], 'lan':0, 'inc':np.nan, 't0':[2452441, 825], 
                    'Mstar':[1.06,0.13], 'plx':[72.45,0.15],'dec':40.4305009,
                    'Mp_is_Mpsini':True},
                'Wittenmyer+ 2007':{'sma':[7.73,0.58], 'Period':[7586,727], 'ecc':[0.005,0.0001], 'Mpsini':[425.87,69.92],
                    'argp':[307, 0.000001], 'lan':0, 'inc':np.nan, 't0':[2452134,146], 
                    'Mstar':[1.06,0.13], 'plx':[72.45,0.15],'dec':40.4305009,
                    'Mp_is_Mpsini':True}
                },
    'HD 219134 h':{
                'Harada+ 2025':{'sma':[2.828,0.0071], 'Period':[2090.0,14], 'ecc':[0.05,0.03], 
                                  'Mpsini':[88,5.4],'argp':[(150+180)%360, 35], 'lan':np.nan, 'inc':np.nan, 
                                't0':[2456221.835261854, 376.6684609391344],'Mstar':[0.81,0.03], #Seager et al. 2021
                                'plx':[153.08, 0.089],'Mp_is_Mpsini':True,
                               'StarTeff':[4700,20], 'Rstar':[0.778,0.005], 'age':[11.00,2.20]},
                'Vogt+ 2015':{'sma':[3.11,0.04], 'Period':[2247,43], 'ecc':[0.06,0.04], 'Mpsini':[0.34*u.Mjup.to(u.Mearth),
                                                                                            0.02*u.Mjup.to(u.Mearth)],
                    'argp':[(215+180)%360, 50], 'lan':0, 'inc':np.nan, 't0':[2448725,356], 
                    'Mstar':[0.794,0.03], 'plx':[153.08,0.09],'dec':57.1696664,
                    'Mp_is_Mpsini':True},
                'Johnson+ 2016':{'sma':[3.00,0.06], 'Period':[2127,63], 'ecc':[0.06,0.0000001], 'Mpsini':[0.256*u.Mjup.to(u.Mearth),
                                                                                            0.029*u.Mjup.to(u.Mearth)],
                    'argp':[(106+180)%360, 73], 'lan':0, 'inc':np.nan, 't0':[2448725,356], 
                    'Mstar':[0.794,0.03], 'plx':[153.08,0.09],'dec':57.1696664,
                    'Mp_is_Mpsini':True},
                },
    '7 CMa c':{
                'Luque+ 2019':{'sma':[2.153,0.08], 'Period':[988,52], 'ecc':[0.08,0.05], 
                                  'Mpsini':[0.87*u.Mjup.to(u.Mearth), 0.06*u.Mjup.to(u.Mearth)],
                                    'argp':[(226.6+180)%360, 40], 'lan':np.nan, 'inc':np.nan, 't0':[2452647.241371557, 72.5826], 
                                    'Mstar':[1.34,0.12], 'plx':[153.08, 0.089],'Mp_is_Mpsini':True,
                               'StarTeff':[4826,55], 'Rstar':[4.87,0.17], 'age':[4.90,1.00]#All star params from Stock+ 2018
                   },
                         },
    '7 CMa b':{
                'Luque+ 2019':{'sma':[1.758,0.024], 'Period':[735.1,14], 'ecc':[0.06,0.03], 
                                  'Mpsini':[1.85*u.Mjup.to(u.Mearth), 0.06*u.Mjup.to(u.Mearth)],
                                                'argp':[(116.4+180)%360, 70], 'lan':np.nan, 'inc':np.nan, 
                                't0':[2452249.3683561394, 175.022756295772], 
                                                'Mstar':[1.34,0.12], 
                                'plx':[153.08, 0.089],
                                                'Mp_is_Mpsini':True,
                               'StarTeff':[4826,55], 'Rstar':[4.87,0.17], 'age':[4.90,1.00]#All star params from Stock+ 2018
                   },
                         },
    'HD 192310 c':{
                'Harada+ 2025':{'sma':[1.183,0.022], 'Period':[546.5,3.4], 'ecc':[0.053,0.061], 
                                  'Mpsini':[14.3,1.2],'argp':[(-170+180)%360, 100], 'lan':np.nan, 'inc':np.nan,
                                't0':[2455224.2024821998, 145.6839], 
                                                'Mstar':[0.85,0.02], 'plx':[113.648,0.1186], 'StarTeff':[5144,50],
                                'Rstar':[0.73,0.02], 'age':[7.81,0.9], #star props from Howard+2011, age Winter+2020
                                                'Mp_is_Mpsini':True
                               },
                         },
    'GJ 687 c':{
                'Benatti+ 2020':{'sma':[1.165,0.023], 'Period':[727.562,12.198], 'ecc':[0.4,0.22], 
                                  'Mpsini':[16,4.1],'argp':[(176+180)%360, 63], 'lan':np.nan, 'inc':np.nan,
                                't0':[2450304.1,292.8], 
                                  'Mstar':[0.40,0.02], 'plx':[219.78,0.03], 'StarTeff':[3561.3,41.8], #Rosenthal+2021
                                'Rstar':[0.397,0.00792], 'age':[], 
                                                'Mp_is_Mpsini':True
                               },
                         },
    'psi1 Dra B b':{
                'Endl+ 2016':{'sma':[4.43,0.04], 'Period':[3117,42], 'ecc':[0.4,0.05], 
                                  'Mpsini':[1.53*u.Mjup.to(u.Mearth), 0.1*u.Mjup.to(u.Mearth)],
                              'argp':[(64+180)%360, 9], 'lan':np.nan, 'inc':np.nan,
                                't0':[2449344, 76], 
                                  'Mstar':[1.19,0.07], 'plx':[45.13,0.049], 'StarTeff':[6212,75], # same ref
                                'Rstar':[1.24,0.08], 'age':[3.3,1.0], 
                                                'Mp_is_Mpsini':True
                               },
                         },
    'HD 128311 c':{
                'Rosenthal+ 2021':{'sma':[1.742,0.02], 'Period':[917.5,1.9], 'ecc':[0.196,0.057], 
                                  'Mpsini':[1033,38],
                              'argp':[(-20+180)%360, 20], 'lan':np.nan, 'inc':np.nan,
                                't0':[2455349.339853104, 49.2761], 
                                  'Mstar':[0.837,0.029], 'plx':[61.21,0.0739], 'StarTeff':[4931.9395,65.975], # same ref
                                'Rstar':[0.78,0.013], 'age':[1.0,0.5], 
                                                'Mp_is_Mpsini':True
                               },
                         },
    'HD 100508 b':{
                'Rosenthal+ 2021':{'sma':[6.11,0.028], 'Period':[5681,42], 'ecc':[0.42,0.02], 
                                  'Mpsini':[1.2*u.Mjup.to(u.Mearth), 0.3*u.Mjup.to(u.Mearth)],
                              'argp':[99.1, 4.2], 'lan':[174,34], 'inc':[124,19],
                                't0':[2447556,87], 
                                  'Mstar':[0.94,0.12], 'plx':[30.51,0.16], 'StarTeff':[5419,112], # same ref
                                'Rstar':[0.78,0.013], 'age':[], '[M/H]':[0.33,0.03], 'age':[3.902,3.334],
                                                'Mp_is_Mpsini':False
                               },
                         },
    'HD 10647 b':{
                'Harada+ 2025':{'sma':[2.073,0.053], 'Period':[991,4.5], 'ecc':[0.16,0.07], 
                                  'Mpsini':[556.0,21],'argp':[(-131+180)%360, 28], 'lan':np.nan, 'inc':np.nan,
                                't0':[2456718.3313935986, 65.9281], 
                                                'Mstar':[1.17,0.19], 'plx':[57.6556,0.07366], 'StarTeff':[6150.97,160],
                                'Rstar':[1.09,0.07], 'age':[1.40,0.90],
                                                'Mp_is_Mpsini':True
                               },
                         },
    'HD 164922 b':{
                'Benatti+ 2020':{'sma':[2.16,0.03], 'Period':[1207,5], 'ecc':[0.08,0.06], 
                                  'Mpsini':[116,12],'argp':[(np.degrees(2.03)+180)%360, np.degrees(4.79)], 'lan':np.nan, 'inc':np.nan,
                                't0':[2450000+7978,26], 
                                                'Mstar':[0.926,0.035], 'plx':[45.422,0.031], 'StarTeff':[5390,30],
                                'Rstar':[0.946,0.021], 'age':[9.580,1.99,1.55], #star props from same ref
                                                'Mp_is_Mpsini':True
                               },
                         },
                         
    '--- Smaller Planets ---':{},
    'Proxima Cen b':{
        'Suarez Mascareno+ 2020':{'sma':[0.04856, 0.00029], 'Period':[11.18418,0.0007], 'ecc':[0.109,0.07], 'Mpsini':[1.16,0.13],
                    'argp':[71.13,41.83], 'lan':0, 'inc':np.nan, 't0':[2458530, 1.3], 'Mstar':[0.1221,0.0022], 'plx':[768.500000,0.203021],
                    'dec':-62.67607,
                    'Mp_is_Mpsini':True}, 
        'Faria+ 2022':{'sma':[0.04856,0.0003], 'Period':[11.1868,0.003], 'ecc':[0.02,0.03], 'Mpsini':[1.07,0.06],
                    'argp':[9,114], 'lan':0, 'inc':np.nan, 't0':[2458515.95, 3.55], 'Mstar':[0.1221,0.0022], 'plx':[768.500000,0.203021],
                    'dec':-62.67607,
                    'Mp_is_Mpsini':True}
                    },
    '61 Vir c':{
        'Vogt+ 2010':{'sma':[0.2175, 0.0001], 'Period':[38.021,0.034], 'ecc':[0.14,0.06], 'Mpsini':[18.2,1.1],
                    'argp':[161, 38], 'lan':0, 'inc':np.nan, 't0':[2453350.5, 4.3], 'Mstar':[0.94, 0.03], 'plx':[117.573,0.237902],
                    'dec':-18.315922,
                    'Mp_is_Mpsini':True}, 
        'Cretignier+ 2023':{'sma':[0.22,0.01], 'Period':[38.09,0.01], 'ecc':[0.06,0.02], 'Mpsini':[17.9,0.8],
                    'argp':[162,22], 'lan':0, 'inc':np.nan, 't0':[2455501.05, 2.34], 
                    'Mstar':[0.94, 0.03], 'plx':[117.573,0.237902],'dec':-18.315922,
                    'Mp_is_Mpsini':True}
                    },
    'HD 192310 b':{
        'Howard+ 2011':{'sma':[0.319,0.005], 'Period':[74.39,0.12], 'ecc':[0.3,0.09], 'Mpsini':[21.6,2.0],
                    'argp':[195,19], 'lan':0, 'inc':np.nan, 't0':[2440000 + 15164.3, 3.6], 
                    'Mstar':[0.8,0.02], 'plx':[113.648,0.118],'dec':-27.03378,
                    'Mp_is_Mpsini':True},
         'Pepe+ 2011':{'sma':[0.32,0.005], 'Period':[74.72,0.1], 'ecc':[0.13,0.04], 'Mpsini':[16.9,0.9],
                    'argp':[353,20], 'lan':0, 'inc':np.nan, 't0':[2455153.58, 4.20], 
                    'Mstar':[0.8,0.02], 'plx':[113.648,0.118],'dec':-27.03378,
                    'Mp_is_Mpsini':True}, 
        'Cretignier+ 2023':{'sma':[0.32,0.01], 'Period':[74.25,0.04], 'ecc':[0.11,0.02], 'Mpsini':[16.1,0.7],
                    'argp':[331,12], 'lan':0, 'inc':np.nan, 't0':[2455521.03, 2.50], 
                    'Mstar':[0.8,0.02], 'plx':[113.648,0.118],'dec':-27.03378,
                    'Mp_is_Mpsini':True},
                    },
    'HD 102365 b':{
        'Howard+ 2011':{'sma':[0.46,0.04], 'Period':[122.1,0.3], 'ecc':[0.34,0.14], 'Mpsini':[16.0,2.6],
                    'argp':[285,22], 'lan':0, 'inc':np.nan, 't0':[2450129, 11], 
                    'Mstar':[0.17,0.02], 'plx':[107.311,0.067],'dec':-40.4985,
                    'Mp_is_Mpsini':True},
                    },
    'GJ 876 e':{
        'Rivera+ 2010':{'sma':[0.3343,0.0013], 'Period':[124.26,0.7], 'ecc':[0.055,0.012], 'Mpsini':[18.2,1.1],
                    'argp':[59,22], 'lan':0, 'inc':59, 't0':[2450486.52, 8.32], 
                    'Mstar':[0.369,0.009], 'plx':[213.867,0.076],'dec':-14.26668,
                    'Mp_is_Mpsini':False},
        'Trifonov+ 2018':{'sma':[0.345,0.002], 'Period':[124.4,0.07], 'ecc':[0.04,0.021], 'Mpsini':[15.43,1.3],
                    'argp':[83,46], 'lan':0, 'inc':59, 't0':[2450494.54, 15.72], 
                    'Mstar':[0.369,0.009], 'plx':[213.867,0.076],'dec':-14.26668,
                    'Mp_is_Mpsini':False},
                    },
    'tau Cet g':{
        'Feng+ 2017':{'sma':[0.133,0.02], 'Period':[20,0.01], 'ecc':[0.06,0.03], 'Mpsini':[1.75,0.25],
                    'argp':[215.34,2.86], 'lan':0, 'inc':np.nan, 't0':[2453591.34, 0.16], 
                    'Mstar':[0.78,0.12], 'plx':[273.8097,0.1701],'dec':-15.933680,
                    'Mp_is_Mpsini':True, 'Note': 'Pub. uncert. represents 1-99\\% intervals'},
        'Feng+ 2017 w/ disk incl':{'sma':[0.133,0.02], 'Period':[20,0.01], 'ecc':[0.06,0.03], 'Mpsini':[1.75,0.25],
                    'argp':[215.34,2.86], 'lan':0, 'inc':[35,10], 't0':[2453591.34, 0.16], 
                    'Mstar':[0.78,0.12], 'plx':[273.8097,0.1701],'dec':-15.933680,
                    'Mp_is_Mpsini':True, 'Note': 'Pub. uncert. represents 1-99\\% intervals, assumes planets are aligned with observed debris disk'}
                    },
    'tau Cet h':{
        'Feng+ 2017':{'sma':[0.243,0.003], 'Period':[49.41,0.1], 'ecc':[0.23,0.15], 'Mpsini':[1.83,0.3],
                    'argp':[187, 40], 'lan':0, 'inc':np.nan, 't0':[2453603.88, 4.47], 
                    'Mstar':[0.78,0.12], 'plx':[273.8097,0.1701],'dec':-15.933680,
                    'Mp_is_Mpsini':True, 'Note': 'Pub. uncert. represents 1-99\\% intervals'},
        'Feng+ 2017 w/ disk incl':{'sma':[0.243,0.003], 'Period':[49.41,0.1], 'ecc':[0.23,0.15], 'Mpsini':[1.83,0.3],
                    'argp':[187, 40], 'lan':0, 'inc':[35,10], 't0':[2453603.88, 4.47],
                    'Mstar':[0.78,0.12], 'plx':[273.8097,0.1701],'dec':-15.933680,
                    'Mp_is_Mpsini':True, 'Note': 'Pub. uncert. represents 1-99\\% intervals, assumes planets are aligned with observed debris disk'}
                    },
    'tau Cet e':{
        'Feng+ 2017':{'sma':[0.538,0.06], 'Period':[162.87,0.5], 'ecc':[0.18,0.14], 'Mpsini':[3.93,0.7],
                    'argp':[202.35, 45], 'lan':0, 'inc':np.nan, 't0':[2453432.45, 26.23], 
                    'Mstar':[0.78,0.12], 'plx':[273.8097,0.1701],'dec':-15.933680,
                    'Mp_is_Mpsini':True, 'Note': 'Pub. uncert. represents 1-99\\% intervals'},
        'Feng+ 2017 w/ disk incl':{'sma':[0.538,0.06], 'Period':[162.87,0.5], 'ecc':[0.18,0.14], 'Mpsini':[3.93,0.7],
                    'argp':[202.35, 45], 'lan':0, 'inc':[35,10], 't0':[2453432.45, 26.23],
                    'Mstar':[0.78,0.12], 'plx':[273.8097,0.1701],'dec':-15.933680,
                    'Mp_is_Mpsini':True, 'Note': 'Pub. uncert. represents 1-99\\% intervals, assumes planets are aligned with observed debris disk'}
                    },
    'tau Cet f':{
        'Feng+ 2017':{'sma':[1.334, 0.017], 'Period':[636.13, 11.7], 'ecc':[0.16,0.07], 'Mpsini':[3.93, 1.0],
                    'argp':[299.74, 40], 'lan':0, 'inc':np.nan, 't0':[2453663.6, 63.8], 
                    'Mstar':[0.78,0.12], 'plx':[273.8097,0.1701],'dec':-15.933680,
                    'Mp_is_Mpsini':True, 'Note': 'Pub. uncert. represents 1-99\\% intervals'},
        'Feng+ 2017 w/ disk incl':{'sma':[1.334, 0.017], 'Period':[636.13, 11.7], 'ecc':[0.16,0.07], 'Mpsini':[3.93, 1.0],
                    'argp':[299.74, 40], 'lan':0, 'inc':[35,10], 't0':[2453663.6, 63.8], 
                    'Mstar':[0.78,0.12], 'plx':[273.8097,0.1701],'dec':-15.933680,
                    'Mp_is_Mpsini':True, 'Note': 'Pub. uncert. represents 1-99\\% intervals, assumes planets are aligned with observed debris disk'}
                    },
    'HD 180617 b':{
        'Burt+ 2021':{'sma':[0.343,0.004], 'Period':[105.911,0.109], 'ecc':[0.101,0.053], 'Mpsini':[12.214,1.05],
                    'argp':[77,51], 'lan':0, 'inc':np.nan, 't0':[2453122.67, 15.50], 
                    'Mstar':[0.484,0.019], 'plx':[169.2,0.05],'dec':5.162976,
                    'Mp_is_Mpsini':True},
        'Kaminski+ 2018':{'sma':[0.3357,0.01], 'Period':[105.9,0.1], 'ecc':[0.16,0.1], 'Mpsini':[12.2,0.01],
                    'argp':[269,45], 'lan':0, 'inc':np.nan, 't0':[2452026.45, 14.65], 
                    'Mstar':[0.484,0.019], 'plx':[169.2,0.05],'dec':5.162976,
                    'Mp_is_Mpsini':True}
                    },
    '--- Roman CGI ---':{},
    'eps Eri b':{
        'Llop-Sayson+ 2026':{'sma':[3.52,0.03], 'Period':[2666,47], 'ecc':[0.04,0.04], 'Mpsini':[0.91*u.Mjup.to(u.Mearth),0.04*u.Mjup.to(u.Mearth)],
                                        'argp':[202,120], 'lan':[-165,5], 'inc':[43,4], 't0':[2457389.0, 1e-12], 
                                        'Mstar':[0.819,0.02], 'plx':[310.576,0.13],
                                        'Mp_is_Mpsini':False, 'StarTeff':[5020.37,57.68], 'Rstar':[0.759,0.0118], 'age':[1.1,0.1]}, #Teff/Rstar from Rosenthal+2021
        'Thompson+ 2025 full soln':{'sma':[3.53,0.04], 'Period':[7.32*u.yr.to(u.d), 0.07*u.yr.to(u.d)], 'ecc':[0.06,0.06], 
                                    'Mpsini':[0.98*u.Mjup.to(u.Mearth),0.9*u.Mjup.to(u.Mearth)],
                    'argp':[83,100], 'lan':[191,9], 'inc':[41,6], 't0':[2458126,1000], 
                    'Mstar':[0.820,0.02], 'plx':[310.577,0.1355],'dec':-9.45816,
                    'Mp_is_Mpsini':False},
                }
}