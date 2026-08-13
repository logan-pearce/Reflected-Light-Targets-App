## List of all RV planets in the exoplanets archive less than 70pc retreived from online interface


```python
import warnings
warnings.filterwarnings('ignore')
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
p = NasaExoplanetArchive.query_criteria(table="pscomppars", select="*", 
                                        where="sy_dist < 70")
p = p.to_pandas()
p = p.sort_values(by=['sy_dist'])
p = p.reset_index(drop = True)
p.to_csv('PS_2026-08-13.csv', index=False)
print(len(p))
```

    1269



```python
p = pd.read_csv('PS_2026-08-13.csv')
p
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>systemid</th>
      <th>sy_name</th>
      <th>objectid</th>
      <th>pl_name</th>
      <th>pl_letter</th>
      <th>hostid</th>
      <th>hostname</th>
      <th>hd_name</th>
      <th>hip_name</th>
      <th>tic_id</th>
      <th>...</th>
      <th>pl_tsmlim</th>
      <th>pl_esm</th>
      <th>pl_esmerr1</th>
      <th>pl_esmerr2</th>
      <th>pl_esm_reflink</th>
      <th>pl_esm_solnid</th>
      <th>pl_esmstr</th>
      <th>pl_esmlim</th>
      <th>sky_coord.ra</th>
      <th>sky_coord.dec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.572780</td>
      <td>alf Cen</td>
      <td>3.18699</td>
      <td>Proxima Cen d</td>
      <td>d</td>
      <td>2.572780</td>
      <td>Proxima Cen</td>
      <td>NaN</td>
      <td>HIP 70890</td>
      <td>TIC 388857263</td>
      <td>...</td>
      <td>0</td>
      <td>2.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;a refstr=CALCULATED_VALUE href=/docs/pscp_cal...</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>217.393466</td>
      <td>-62.676182</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.572780</td>
      <td>alf Cen</td>
      <td>3.11922</td>
      <td>Proxima Cen b</td>
      <td>b</td>
      <td>2.572780</td>
      <td>Proxima Cen</td>
      <td>NaN</td>
      <td>HIP 70890</td>
      <td>TIC 388857263</td>
      <td>...</td>
      <td>0</td>
      <td>0.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;a refstr=CALCULATED_VALUE href=/docs/pscp_cal...</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>217.393466</td>
      <td>-62.676182</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.583550</td>
      <td>Barnard's star</td>
      <td>3.20280</td>
      <td>Barnard d</td>
      <td>d</td>
      <td>2.583550</td>
      <td>Barnard's star</td>
      <td>NaN</td>
      <td>HIP 87937</td>
      <td>TIC 325554331</td>
      <td>...</td>
      <td>0</td>
      <td>14.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;a refstr=CALCULATED_VALUE href=/docs/pscp_cal...</td>
      <td>NaN</td>
      <td>14</td>
      <td>0</td>
      <td>269.448614</td>
      <td>4.737981</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.583550</td>
      <td>Barnard's star</td>
      <td>3.19959</td>
      <td>Barnard b</td>
      <td>b</td>
      <td>2.583550</td>
      <td>Barnard's star</td>
      <td>NaN</td>
      <td>HIP 87937</td>
      <td>TIC 325554331</td>
      <td>...</td>
      <td>0</td>
      <td>10.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;a refstr=CALCULATED_VALUE href=/docs/pscp_cal...</td>
      <td>NaN</td>
      <td>11</td>
      <td>0</td>
      <td>269.448614</td>
      <td>4.737981</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.583550</td>
      <td>Barnard's star</td>
      <td>3.20281</td>
      <td>Barnard e</td>
      <td>e</td>
      <td>2.583550</td>
      <td>Barnard's star</td>
      <td>NaN</td>
      <td>HIP 87937</td>
      <td>TIC 325554331</td>
      <td>...</td>
      <td>0</td>
      <td>2.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;a refstr=CALCULATED_VALUE href=/docs/pscp_cal...</td>
      <td>NaN</td>
      <td>3</td>
      <td>0</td>
      <td>269.448614</td>
      <td>4.737981</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1264</th>
      <td>1.576689</td>
      <td>HD 35843</td>
      <td>3.16899</td>
      <td>HD 35843 c</td>
      <td>c</td>
      <td>2.576704</td>
      <td>HD 35843</td>
      <td>HD 35843</td>
      <td>HIP 25359</td>
      <td>TIC 7422496</td>
      <td>...</td>
      <td>0</td>
      <td>0.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;a refstr=CALCULATED_VALUE href=/docs/pscp_cal...</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>81.348892</td>
      <td>-44.431285</td>
    </tr>
    <tr>
      <th>1265</th>
      <td>1.585780</td>
      <td>HD 179079</td>
      <td>3.11169</td>
      <td>HD 179079 b</td>
      <td>b</td>
      <td>2.585780</td>
      <td>HD 179079</td>
      <td>HD 179079</td>
      <td>HIP 94256</td>
      <td>TIC 48917782</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>287.790373</td>
      <td>-2.638770</td>
    </tr>
    <tr>
      <th>1266</th>
      <td>1.556860</td>
      <td>HD 72892</td>
      <td>3.11951</td>
      <td>HD 72892 b</td>
      <td>b</td>
      <td>2.556860</td>
      <td>HD 72892</td>
      <td>HD 72892</td>
      <td>HIP 42098</td>
      <td>TIC 405344271</td>
      <td>...</td>
      <td>0</td>
      <td>39.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;a refstr=CALCULATED_VALUE href=/docs/pscp_cal...</td>
      <td>NaN</td>
      <td>39</td>
      <td>0</td>
      <td>128.719555</td>
      <td>-14.457431</td>
    </tr>
    <tr>
      <th>1267</th>
      <td>1.574319</td>
      <td>TOI-1648</td>
      <td>3.14478</td>
      <td>TOI-1648 b</td>
      <td>b</td>
      <td>2.574334</td>
      <td>TOI-1648</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>TIC 376353509</td>
      <td>...</td>
      <td>0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;a refstr=CALCULATED_VALUE href=/docs/pscp_cal...</td>
      <td>NaN</td>
      <td>5</td>
      <td>0</td>
      <td>45.401179</td>
      <td>69.229923</td>
    </tr>
    <tr>
      <th>1268</th>
      <td>1.576610</td>
      <td>eps CrB</td>
      <td>3.11552</td>
      <td>eps CrB b</td>
      <td>b</td>
      <td>2.576610</td>
      <td>eps CrB</td>
      <td>HD 143107</td>
      <td>HIP 78159</td>
      <td>TIC 356000102</td>
      <td>...</td>
      <td>0</td>
      <td>5.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;a refstr=CALCULATED_VALUE href=/docs/pscp_cal...</td>
      <td>NaN</td>
      <td>6</td>
      <td>0</td>
      <td>239.396506</td>
      <td>26.877617</td>
    </tr>
  </tbody>
</table>
<p>1269 rows × 705 columns</p>
</div>



## If radius is missing
Use mass-radius relation and minimum mass to estimate radius.


```python
def PlanetMass2Radius(M):
    ''' Theoretical mass-radius relation for planets and brown dwarfs by Jared
        taken from 
        https://jaredmales.github.io/mxlib-doc/group__planets.html#ga4b350ecfdeaca1bedb897db770b09789
    '''
    try:
        M = M.to(u.Mearth)
        M = M.value
    except:
        pass
    
    if M < 4.1:
        R = M**(1/3)
        
    if M >= 4.1 and M < 15.84:
        R = 0.62 * M**(0.67)
        
    if M >= 15.84 and M < 3591.1:
        coeff = [14.0211, -44.8414, 53.6554, -25.3289, 5.4920, -0.4586]
        power = [0, 1, 2, 3, 4, 5]
        R = 0
        for i in range(6):
            R += coeff[i] * (np.log10(M)**power[i])
            
    if M >= 3591.1:
        R = 32.03 * M**(-1/8)
        
    return R
    
p['M2R infered radius [Rearth]'] = np.nan
for i in range(len(p)):
    if np.isnan(p.loc[i]['pl_rade']):
        if not np.isnan(p.loc[i]['pl_bmasse']):
            p['M2R infered radius [Rearth]'][i] = PlanetMass2Radius(p.loc[i]['pl_bmasse'])
            
```

## SMA:
drop ones without sma 


```python
ind = np.where(np.isnan(p['pl_orbsmax']))[0]
p = p.drop(ind)
p = p.reset_index(drop = True)
```

## Get Spectral Type/Teff if missing:

#### If there is no SpT, use Teff to estimate.  If there is no Teff, use SpT to estimate.  Using the Mamjek table
https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt


```python
# put spectral types into numeric scheme X.YY where X is a number corresponding to letter type with 0 == O and 6 ==M,
# and .YY is the number converted to decimal.

def GetSpTNumber(j):
    sptletters = ['O','B','A','F','G','K','M','L','T','Y']
    sptnumbers = [0,1,2,3,4,5,6,7,8,9]
    ind = np.where([x['SpT'][j][0]==sptletters[i] for i in range(len(sptletters))])
    n = sptnumbers[ind[0][0]]
    n += float(x['SpT'][j][1])/10
    if '.' in x['SpT'][j]:
        n += 0.05
    return n

# Convert SpT to numbers to allow for interpolation:
x = pd.read_table('EEM_dwarf_UBVIJHK_colors_Teff.txt', delim_whitespace=True, comment='#',
                 nrows=118)
x['SpTNumbers'] = np.nan
for j in range(len(x)):
    x['SpTNumbers'][j] = GetSpTNumber(j)

```


```python
# Create SpT/Teff lookup splines:
from scipy.interpolate import UnivariateSpline
GetTeffSpl = UnivariateSpline(x['SpTNumbers'], x['Teff'])
GetSpTSpl = UnivariateSpline(x['Teff'][::-1],x['SpTNumbers'][::-1])
```


```python
# Now turn star SpT into numbers:
def GetSpTNumber(j):
    sptletters = ['O','B','A','F','G','K','M','L','T','Y']
    sptnumbers = [0,1,2,3,4,5,6,7,8,9]
    try:
        ind = np.where([p['st_spectype'][j][0]==sptletters[i] for i in range(len(sptletters))])
    except TypeError:
        return np.nan
    try:
        n = sptnumbers[ind[0][0]]
    except IndexError:
        pass
    try:
        n += float(p['st_spectype'][j][1])/10
        if '.5' in p['st_spectype'][j]:
            n += 0.05
        return n
    except:
        pass


p['SpTNumber'] = np.nan
for j in range(len(p)):
    if p['st_spectype'][j] == '':
        pass
    else:
        p['SpTNumber'][j] = GetSpTNumber(j)
```

#### If there's no SpT infer it from Teff table:


```python
p['Inferred SpT from Teff'] = np.nan
ind = np.where(np.isnan(p['SpTNumber']))[0]

for i in ind:
    #print(i,p['SpTNumber'][i],p['st_teff'][i])
    p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])  
    #print(p['Inferred SpT from Teff'][i])
    
ind = np.where(np.isnan(p['SpTNumber']))[0]
for i in ind:
    p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
print(np.where(np.isnan(p['SpTNumber']))[0])
```

    [199 721]


    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: FutureWarning: ChainedAssignmentError: behaviour will change in pandas 3.0!
    You are setting values through chained assignment. Currently this works in certain cases, but when using Copy-on-Write (which will become the default behaviour in pandas 3.0) this will never work to update the original DataFrame or Series, because the intermediate object on which we are setting values will behave as a copy.
    A typical example is when you are setting values in a column of a DataFrame, like:
    
    df["col"][row_indexer] = value
    
    Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['Inferred SpT from Teff'][i] = GetSpTSpl(p['st_teff'][i])
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]
    /var/folders/jg/989ldvh91l31v90slbnhd8x80000gr/T/ipykernel_28555/1675411421.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      p['SpTNumber'][i] = p['Inferred SpT from Teff'][i]


#### If no Teff infer it from SpT table:


```python
p['Inferred Teff from SpT'] = np.nan
p['StarTeff'] = p['st_teff']

ind = np.where(np.isnan(p['st_teff']))

for i in ind:
    p['Inferred Teff from SpT'][i] = GetTeffSpl(p['SpTNumber'][i])  
    p['StarTeff'][i] = GetTeffSpl(p['SpTNumber'][i]) 
```

### Compile radius:


```python
p['PlanetRadiuse'] = p['pl_rade'].copy()
ind = np.where(np.isnan(p['pl_rade']))[0]
p['PlanetRadiuse'][ind] = p['M2R infered radius [Rearth]'][ind]
np.where(np.isnan(p['PlanetRadiuse']))
```




    (array([], dtype=int64),)



## Star properties

#### If missing radius get from Mamjek table using Teff:


```python
x = pd.read_table('EEM_dwarf_UBVIJHK_colors_Teff.txt', delim_whitespace=True, comment='#',
                 nrows=118)

for i in range(len(x)):
    if x['R_Rsun'][i] == '...':
        x['R_Rsun'][i] = np.nan
    x['R_Rsun'][i] = float(x['R_Rsun'][i])

from scipy.interpolate import UnivariateSpline
GetStarRadSpl = UnivariateSpline(x['Teff'][:110][::-1], x['R_Rsun'][:110][::-1])

```


```python
p['StarRad'] = p['st_rad'].copy()

p['Inferred StarRad from Teff'] = np.nan
ind = np.where(np.isnan(p['st_rad']))[0]
for i in ind:
    p['Inferred StarRad from Teff'][i] = GetStarRadSpl(p['StarTeff'][i])
    p['StarRad'][i] = GetStarRadSpl(p['StarTeff'][i])
```


```python
#p.to_csv('Refl-light-target-list.csv',index=False)
```

# Compute typical/maximum separation


```python
orbits = p[['pl_name','pl_orbsmax','pl_orbper','pl_orbincl','pl_orbeccen','pl_bmasse','pl_bmassj','st_mass', 
             'sy_dist', 'PlanetRadiuse', 'pl_orblper','StarTeff', 'st_logg', 'st_met', 'StarRad','sy_imag',
            'sy_gaiamag','rastr','ra','decstr','dec','SpTNumber']]
```

## Drop planets missing necessary information:


```python
print('Missing sma:',np.where(np.isnan(orbits['pl_orbsmax']))[0])
orbits = orbits.drop(np.where(np.isnan(orbits['pl_orbsmax']))[0])
orbits = orbits.reset_index(drop=True)
print('Missing star mass:',np.where(np.isnan(orbits['st_mass']))[0])
orbits = orbits.drop(np.where(np.isnan(orbits['st_mass']))[0])
orbits = orbits.reset_index(drop=True)
print('Missing period:',np.where(np.isnan(orbits['pl_orbper']))[0])
orbits = orbits.drop(np.where(np.isnan(orbits['pl_orbper']))[0])
orbits = orbits.reset_index(drop=True)
print('Missing eccen:',np.where(np.isnan(orbits['pl_orbeccen']))[0])
orbits = orbits.drop(np.where(np.isnan(orbits['pl_orbeccen']))[0])
orbits = orbits.reset_index(drop=True)
print('Missing argp:',np.where(np.isnan(orbits['pl_orblper']))[0])
orbits = orbits.drop(np.where(np.isnan(orbits['pl_orblper']))[0])
orbits = orbits.reset_index(drop=True)
```

    Missing sma: []
    Missing star mass: []
    Missing period: [ 119  144  201  264  279  298  335  350  363  593  594  595  596  599
      714  728  735  820  868  881  917  919  958 1025 1145 1167 1209]
    Missing eccen: [   8   10   40   41   75   76   77   85   93  102  103  104  110  119
      134  147  151  152  178  179  190  214  233  234  242  274  279  289
      326  343  354  369  370  371  376  377  413  461  464  465  479  510
      515  516  517  528  533  546  547  548  549  550  551  556  557  560
      614  647  648  674  682  736  739  742  748  764  770  777  788  817
      825  826  848  850  865  879  880  897  914  923  939  980  991 1004
     1040 1050 1051 1061 1080 1093 1099 1114 1115 1118 1119 1120 1132 1137
     1142 1171 1176 1178 1179 1181 1182]
    Missing argp: [   9   27   41   42   50   51   55   56   80   97  110  117  144  145
      146  155  174  177  186  187  237  249  250  251  254  264  280  281
      293  294  295  296  305  310  311  312  313  325  356  357  365  373
      375  394  395  396  413  414  430  431  436  437  438  440  441  454
      455  456  465  499  550  552  565  573  574  575  576  577  578  582
      611  613  614  653  654  655  676  677  678  689  705  707  727  754
      770  782  795  796  848  849  854  867  877  890  956  957  975  990
      998 1018 1019 1020 1021 1022 1046 1052 1075 1078]


## Computing contrast and phase:

From Cahoy 2010 Eqn 1:

$$ C(\alpha) = A_g(\lambda) \left(\frac{R_p}{r}\right)^2 \left[\frac{\sin\alpha + (\pi - \alpha)\cos\alpha}{\pi} \right]$$
where<br>
$C(\alpha)$ is planet-star contrast<br>
$ A_g(\lambda)$ is geometric albedo<br>
$R_p$ is planet radius<br>
$r$ is planet-star true separation (in the orbit plane)<br>

And phase as a function of orbital elements is given by:<br>

$$\alpha = \cos^{-1} \left(\sin(i) \;\times\; \sin(\theta + \omega_p)\right)$$
where<br>
$\omega_p$ is argument of periastron<br>
$i$ is inclination, with i=90 being edge on and i = 0,180 being face on<br>
$\theta$ is the true anomaly with
$$\theta = 2 \tan^{-1} \left(\sqrt{\frac{1+e}{1-e}} \tan(E/2) \right)$$
where<br>
$e$ is the eccentricity<br>
$E$ is the eccentricity anomaly<br>
with
$$M = E - e \sin E$$
$$M = 2\pi \frac{\Delta t}{P}$$
where<br>
$M$ is the mean anomaly<br>
$\Delta t$ is the time since periastron passage<br>
$P$ is the orbital period
<br><br>


```python
def eccentricity_anomaly(E,e,M):
    '''Eccentric anomaly function'''
    import numpy as np
    return E - (e*np.sin(E)) - M

def danby_solve(f, M0, e, h, maxnum=50):
    ''' Newton-Raphson solver for eccentricity anomaly based on "Danby" method in 
        Wisdom textbook
    Inputs: 
        f (function): function to solve (transcendental ecc. anomaly function)
        M0 (float): mean anomaly
        e (float): eccentricity
        h (float): termination criteria for solver
        maxnum (int): if it takes more than maxnum iterations,
            use the Mikkola solver instead.
    Returns: nextE (float): converged solution for eccentric anomaly
        Written by Logan Pearce, 2020
    '''
    import numpy as np
    from myastrotools.tools import eccentricity_anomaly
    #f = eccentricity_anomaly
    k = 0.85
    E0 = M0 + np.sign(np.sin(M0))*k*e
    lastE = E0
    nextE = lastE + 10* h 
    number=0
    delta_D = 1
    while (delta_D > h) and number < maxnum+1: 
        fx = f(nextE,e,M0) 
        fp = (1.-e*np.cos(lastE)) 
        fpp = e*np.sin(lastE)
        fppp = e*np.cos(lastE)
        lastE = nextE
        delta_N = -fx / fp
        delta_H = -fx / (fp + 0.5*fpp*delta_N)
        delta_D = -fx / (fp + 0.5*fpp*delta_H + (1./6)*fppp*delta_H**2)
        nextE = lastE + delta_D
        number=number+1
        if number >= maxnum:
            from myastrotools.tools import mikkola_solve
            nextE = mikkola_solve(M0,e)
    return nextE

def ComputeFluxRatio(Rp, sep_in_orbit_plane, alpha, Ag = 0.5):
    ''' For a single planet compute planet/star flux ratio using Cahoy 2010 eqn 1
    and https://ui.adsabs.harvard.edu/abs/2017ApJ...844...89C/abstract
    
    Args:
        Rp (astropy units object): planet radius
        sep (astropy units object): planet-star separation in the plane of the orbit (true separation)
        alpha (flt): phase angle in degrees
        Ag (flt): geometric albedo

    Returns:
        flt: planet-star contrast
    '''
    alpha = np.radians(alpha)
    angleterm = (np.sin(alpha) + (np.pi - alpha)*np.cos(alpha)) / np.pi
    Rp = Rp.to(u.km)
    sep_in_orbit_plane = sep_in_orbit_plane.to(u.km)
    C = Ag * ((Rp / sep_in_orbit_plane)**2) * angleterm
    return C


def GetPhaseAngle(MeanAnom, ecc, inc, argp):
    ''' Function for returning observed phase angle given orbital elements
    Args:
        MeanAnom (flt): Mean anomly in radians, where MeanAnom = orbit fraction*2pi, or M=2pi * time/Period
        ecc (flt): eccentricity, defined on [0,1)
        inc (flt): inclination in degrees, where inc = 90 is edge on, inc = 0 or 180 is face on orbit
        argp (flt): argument of periastron in degrees, defined on [0,360)
        
    Returns:
        flt: phase angle in degrees
    Written by Logan Pearce, 2023
    '''
    import numpy as np
    inc = np.radians(inc)
    argp = np.radians(argp)
    EccAnom = danby_solve(eccentricity_anomaly, MeanAnom, ecc, 0.001, maxnum=50)
    TrueAnom = 2*np.arctan( np.sqrt( (1+ecc)/(1-ecc) ) * np.tan(EccAnom/2) )
    Alpha = np.arccos( np.sin(inc) * np.sin(TrueAnom + argp) )
    return np.degrees(Alpha)
```

## Function for computing sepatation, contrast, and phase along a keplerian orbit


```python
from projecc import KeplerianToCartesian, KeplersConstant, DanbySolve, EccentricityAnomaly

def GetOrbitPlaneOfSky(sma,ecc,inc,argp,lon,meananom,kep):
    ''' For a value fo sma, ecc, inc, argp, lan, and mass, compute the position in the sky plane for one or
    an array of mean anomaly values past periastron.

    args:
        sma [astropy unit object]: semi-major axis in au
        ecc [flt]: eccentricity
        inc [flt]: inclination in degrees
        argp [flt]: argument of periastron in degrees
        lon [flt]: longitude of nodes in degrees
        meananom [flt or arr]: a single value or array of values for the mean anomaly in radians at which 
            to compute positions
        kep [astropy unit object]: value of Kepler's constant for the system
    
    returns:
        flt or arr: X value of position, where +x corresponds to +Declination
        flt or arr: Y value, where +Y corresponds to +Right Ascension
        flt or arr: Z value, where +Z corresponds to towards the observer
    '''
    pos, vel, acc = KeplerianToCartesian(sma,ecc,inc,argp,lon,meananom,kep)
    return pos[0].value, pos[1].value, pos[2].value

def GetOrbitPlaneOfOrbit(sma,ecc,meananom,kep):
    ''' For a value fo sma, ecc, and mass, compute the position in the orbit plane for one or
    an array of mean anomaly values past periastron.

    args:
        sma [astropy unit object]: semi-major axis in au
        ecc [flt]: eccentricity
        meananom [flt or arr]: a single value or array of values for the mean anomaly in radians at which 
            to compute positions
        kep [astropy unit object]: value of Kepler's constant for the system
    
    returns:
        flt or arr: x value of position, where +x corresponds to semi-major axis towards periastron
        flt or arr: y value, where +y corresponds to semi-minor axis counterclockwise perpendiculat to +x
        flt or arr: z value, where +z corresponds to angular momentum vector for right handed system
    '''
    import numpy as np
    import astropy.units as u
    E = DanbySolve(EccentricityAnomaly, meananom, ecc, 0.001)
    return (sma*(np.cos(E) - ecc)).value, (sma*np.sqrt(1-ecc**2)*np.sin(E)).value, 0

def ComputeThings(M1, M2, sma, ecc, inc, argp, lon, Rp, Ag=0.45):
    # compute kepler's constant:
    kep = KeplersConstant(M1,M2)
    # empt arrays to store things:
    xs = []
    ys = []
    zs = []
    truexs = []
    trueys = []
    truezs = []
    alphas = []
    n=500
    # array of mean anomalies for each point on the orbit:
    meananom = np.linspace(0,2*np.pi,n)
    # for each point:
    for j in range(len(meananom)):
        # get x,y,z postion in plane of orbit and plane of sky
        pos_planeofsky = GetOrbitPlaneOfSky(sma,ecc,inc,argp,lon,meananom[j],kep)
        pos_planeoforbit = GetOrbitPlaneOfOrbit(sma,ecc,meananom[j],kep)

        xs.append(pos_planeofsky[0])
        ys.append(pos_planeofsky[1])
        zs.append(pos_planeofsky[2])
        truexs.append(pos_planeoforbit[0])
        trueys.append(pos_planeoforbit[1])
        truezs.append(pos_planeoforbit[2])
        # compute phase at that point:
        alpha = GetPhaseAngle(meananom[j], ecc, inc, argp)
        alphas.append(alpha)

    # turn into arrays for mathing:
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    truexs = np.array(truexs)
    trueys = np.array(trueys)
    truezs = np.array(truezs)
    alphas = np.array(alphas)

    # get separation in plane of orbit and sky at each orbit point:
    sep_planeoforbit = np.sqrt(truexs**2 + trueys**2)
    sep_planeofsky = np.sqrt(xs**2 + ys**2)
    # get contrasts at each orbit point:
    contrasts = np.array([ComputeFluxRatio(Rp, sep_planeoforbit[i]*u.au, alphas[i], Ag = Ag) for i in range(len(alphas))])
    
    return xs, ys, zs, truexs, trueys, truezs, sep_planeoforbit, sep_planeofsky, alphas, contrasts


```


```python
### Compite phase and contrast at max projected separation:b

from projecc import update_progress
maxprojseps = []
phases = []
conts = []
seps_planeofsky = []
seps_planeoforbit = []
allconts = []
allphases = []

for i in range(len(orbits)):
    # for each planet:
    p = orbits.loc[i]
    # get the stuff:
    M1 = p['st_mass'].item()*u.Msun
    M2 = p['pl_bmasse'].item()*u.Mearth
    sma = p['pl_orbsmax'].item()*u.au
    ecc = p['pl_orbeccen'].item()
    inc = p['pl_orbincl'].item() # deg
    if np.isnan(inc):
        inc = 60
    argp = p['pl_orblper'].item() # deg
    ### Assuming argp refers to the star, appliy 180 deg offset for planet argp:
    argp = (argp + 180) % 360
    lon = 0
    Rp = p['PlanetRadiuse'].item()*u.Rearth
    # compute the things
    xs, ys, zs, truexs, trueys, truezs, sep_planeoforbit, sep_planeofsky, alphas, contrasts = ComputeThings(M1,M2,sma,ecc,inc,argp,lon,Rp,Ag=0.45)
    # find the max projected separation and it's index:
    maxprojsep = np.max(sep_planeofsky)
    maxprojsepind = np.where(sep_planeofsky == maxprojsep)[0]
    # record the results:
    maxprojseps.append(sep_planeofsky[maxprojsepind])
    phases.append(alphas[maxprojsepind])
    conts.append(contrasts[maxprojsepind])
    seps_planeofsky.append(sep_planeofsky)
    seps_planeoforbit.append(sep_planeoforbit)
    allconts.append(contrasts)
    allphases.append(alphas)
    update_progress(i,len(orbits))
    
phases = np.array([phases[i][0] for i in range(len(phases))])
conts = np.array([conts[i][0] for i in range(len(conts))])
maxprojseps = np.array([maxprojseps[i][0] for i in range(len(maxprojseps))])
```

    100.0% (975 of 976): |####################|  


```python
orbits['MaxProjectedSeparation_au'] = maxprojseps
orbits['MaxProjectedSeparation_mas'] = (orbits['MaxProjectedSeparation_au']/orbits['sy_dist'])*1000
orbits['PhaseAtMaxProj'] = phases
orbits['ContrastAtMaxProj'] = conts

# Check to make sure there is a correlation between eccentricity and phase
%matplotlib inline
plt.hist2d(orbits['pl_orbeccen'],orbits['PhaseAtMaxProj'],bins=20,cmin=1)
plt.colorbar()
plt.xlabel('ecc')
plt.ylabel('phase [deg]')
plt.show()
```


    
![png](output_33_0.png)
    



```python
lod_elt = (0.2063 * 0.8 / 39) * 1000
lod_gmagaox = (0.2063 * 0.8 / 25.4) * 1000
lod_magaox = (0.2063 * 0.8 / 6.5) * 1000
orbits['MaxProjectedSeparation_lod_elt'] = orbits['MaxProjectedSeparation_mas']/lod_elt
orbits['MaxProjectedSeparation_lod_gmagaox'] = orbits['MaxProjectedSeparation_mas']/lod_gmagaox
orbits['MaxProjectedSeparation_lod_magaox'] = orbits['MaxProjectedSeparation_mas']/lod_magaox
```


```python
### Save the whole orbit points in the db as well for good measure:
import warnings
warnings.filterwarnings('ignore')
orbits['SepsInPlaneOfSky_au'] = pd.Series(np.nan, dtype='object')
orbits['SepsInPlaneOfOrbit_au'] = pd.Series(np.nan, dtype='object')
orbits['Contrasts'] = pd.Series(np.nan, dtype='object')
orbits['Phases'] = pd.Series(np.nan, dtype='object')
for i in range(len(orbits)):
    orbits['SepsInPlaneOfSky_au'][i] = seps_planeofsky[i]
    orbits['SepsInPlaneOfOrbit_au'][i] = seps_planeoforbit[i]
    orbits['Contrasts'][i] = allconts[i]
    orbits['Phases'][i] = allphases[i]

orbits['SepsInPlaneOfSky_mas'] = pd.Series(np.nan, dtype='object')
orbits['SepsInPlaneOfSky_lod_magaox'] = pd.Series(np.nan, dtype='object')
orbits['SepsInPlaneOfSky_lod_gmagaox'] = pd.Series(np.nan, dtype='object')
orbits['SepsInPlaneOfSky_lod_elt'] = pd.Series(np.nan, dtype='object')
for i in range(len(orbits)):
    orbits['SepsInPlaneOfSky_mas'][i] = (orbits['SepsInPlaneOfSky_au'][i]/orbits['sy_dist'][i])*1000
    orbits['SepsInPlaneOfSky_lod_magaox'][i] = orbits['SepsInPlaneOfSky_mas'][i]/lod_magaox
    orbits['SepsInPlaneOfSky_lod_gmagaox'][i] = orbits['SepsInPlaneOfSky_mas'][i]/lod_gmagaox
    orbits['SepsInPlaneOfSky_lod_elt'][i] = orbits['SepsInPlaneOfSky_mas'][i]/lod_elt
```


```python
orbits.to_csv('Target-list-with-orbital-params.csv', index=False)
```

## "Typical" sep/cont:
A weighted average of the orbital separation weigthed by contrast.

### Weights:
$$S/N \propto Cp$$

"Typical" = contrast weighted average

For all points of orbit outside IWA:


```python
# pick a planet:
i = 20

iwa = np.where(orbits['SepsInPlaneOfSky_lod_gmagaox'][i] > 2)
seps = orbits['SepsInPlaneOfSky_lod_gmagaox'][i][iwa]
conts = orbits['Contrasts'][i][iwa]
phases = orbits['Phases'][i][iwa]

typical_sep_contsq = np.sum(seps*(conts**2))/np.sum((conts**2))
typical_cont = np.sum(conts*(conts**2))/np.sum((conts**2))
typical_phase = np.sum(phases*(conts**2))/np.sum((conts**2))

typical_sep_contsq
```




    nan




```python
### Examples of typical separation:

p = orbits.loc[i]
# get the stuff:
M1 = p['st_mass'].item()*u.Msun
M2 = p['pl_bmasse'].item()*u.Mearth
sma = p['pl_orbsmax'].item()*u.au
ecc = p['pl_orbeccen'].item()
inc = p['pl_orbincl'].item() # deg
if np.isnan(inc):
    inc = 60
argp = p['pl_orblper'].item() # deg
argp = (argp + 180) % 360
lon = 0
Rp = p['PlanetRadiuse'].item()*u.Rearth
xs, ys, zs, truexs, trueys, truezs, sep_planeoforbit, sep_planeofsky, alphas, contrasts = ComputeThings(M1,M2,sma,ecc,inc,argp,lon,Rp,Ag=0.45)

%matplotlib inline
fig, ax = plt.subplots(figsize=(8,8))
ps = ax.scatter(ys,xs, c=alphas, cmap="seismic", alpha=0.5)

nodes_idx = np.where(np.sign(zs[:-1]) != np.sign(zs[1:]))[0] + 1
ax.scatter(ys[nodes_idx],xs[nodes_idx], marker='x',color='grey', s=100)
ax.plot(ys[nodes_idx],xs[nodes_idx], color='grey', ls=':', label='line of nodes')

sortseps = np.argsort(orbits['SepsInPlaneOfSky_lod_gmagaox'][i])
ax.scatter(ys[sortseps[:2]],xs[sortseps[:2]], marker='x',color='orange', s=100, label='closest proj sep')
string = [int(orbits['Phases'][i][sortseps[:2]][j]) for j in range(2)]
for j in range(2):
    ax.annotate('Phase='+str(string[j]),xy = (ys[sortseps[j]],xs[sortseps[j]]),
           xytext = (10,5), textcoords='offset points')
    
    
isclose = np.where(np.isclose(typical_sep_contsq,orbits['SepsInPlaneOfSky_lod_gmagaox'][i],rtol=5e-03))[0]
isclose = [60,190,310,439]
ax.scatter(ys[isclose],xs[isclose], marker='x',color='purple', s=100, label='typical proj sep')
string2 = [int(orbits['Phases'][i][isclose][j]) for j in range(4)]
for j in range(4):
    ax.annotate('Phase='+str(string2[j]),xy = (ys[isclose[j]],xs[isclose[j]]),
           xytext = (10,5), textcoords='offset points')

ax.scatter(0,0,marker='*',color='orange',s=100)
ax.annotate('e='+str(ecc),xy = (0.05,0.05), fontsize= 20, xycoords='axes fraction')
ax.annotate('inc='+str(inc),xy = (0.05,0.09), fontsize= 20, xycoords='axes fraction')


ax.set_xlabel('RA [au]')
ax.set_ylabel('Dec [au]')
# ax.set_xlim(-0.08,0.08)
# ax.set_ylim(-0.08,0.08)
ax.invert_xaxis()
ax.set_aspect('equal')
ax.grid(ls=':')
ax.legend(fontsize=15)
cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])
fig.colorbar(ps, cax=cbar_ax)
cbar_ax.set_ylabel('Phase [deg]')
#plt.savefig('example-typical-sep-orbit.png')
```




    Text(0, 0.5, 'Phase [deg]')




    
![png](output_39_1.png)
    



```python
#i = np.where(orbits['pl_name'] == 'GJ 876 b')[0][0]

%matplotlib inline
plt.figure(figsize=(12,7))
plt.subplot(121)
plt.plot(range(500),orbits['SepsInPlaneOfSky_lod_gmagaox'][i], label='separation')
#plt.axhline(y=typical_sep_cont,color='orange')
plt.axhline(y=typical_sep_contsq,color='purple')
plt.plot(np.arange(0,500,1)[iwa],orbits['SepsInPlaneOfSky_lod_gmagaox'][i][iwa],ls='--',label='sep outside iwa')
plt.fill_between(range(500), 2, where=ys<=2, interpolate=True, color='grey', alpha = 0.1)


plt.scatter(isclose,orbits['SepsInPlaneOfSky_lod_gmagaox'][i][isclose], marker='x',color='purple', s=100, 
            label='typical separation',zorder=10)

plt.legend(loc=(0.1,0.05),fontsize=15)
plt.ylim(bottom=1.5)
plt.ylabel('Sep in GMagAO-X LOD in i')


plt.subplot(122)
plt.plot(range(500),orbits['Contrasts'][i])
#plt.axhline(y=typical_cont_cont,color='orange')
#plt.axhline(y=typical_cont_contsq,color='purple')
#plt.axhline(y=typical_cont2,color='grey', label='typical contrast')
plt.plot(np.arange(0,500,1)[iwa],orbits['Contrasts'][i][iwa],ls='--')
plt.scatter(sortseps[:2],orbits['Contrasts'][i][sortseps[:2]], marker='x',color='orange', s=100, 
            label='closest proj sep',zorder=10)

plt.scatter(isclose,orbits['Contrasts'][i][isclose], marker='x',color='purple', s=100, 
            zorder=10, label='loc of typical separation')

yoff = [-8,3]
xoff = [-100,20]
for j in range(2):
    plt.annotate('Phase='+str(string[j]),xy = (sortseps[j],orbits['Contrasts'][i][sortseps[j]]),
           xytext = (xoff[j],yoff[j]), textcoords='offset points', fontsize=20, color='orange')

yoff = [0,5,-8,0]
xoff = [0,10,-100,0]
for j in range(4):
    if j == 1 or j == 2:
        plt.annotate('Phase='+str(string2[j]),xy = (isclose[j],orbits['Contrasts'][i][isclose[j]]),
           xytext = (xoff[j],yoff[j]), textcoords='offset points', fontsize=20, color='purple')
        
phaseisclosetotypicalcont = np.where(np.isclose(typical_cont,orbits['Contrasts'][i],atol=1e-9))[0]

plt.scatter(phaseisclosetotypicalcont,orbits['Contrasts'][i][phaseisclosetotypicalcont],
           marker='x',color='lightseagreen',s=100, label='typical phase')
string3 = [int(orbits['Phases'][i][phaseisclosetotypicalcont][j]) for j in range(2)]
yoff = [-25,-20]
xoff = [10,20]
for j in range(1):
    plt.annotate('Phase='+str(string3[j]),xy = (phaseisclosetotypicalcont[j],
                                                orbits['Contrasts'][i][phaseisclosetotypicalcont[j]]),
           xytext = (xoff[j],yoff[j]), textcoords='offset points', fontsize=20, color='lightseagreen')


plt.ylabel('Contrast')
plt.gca().set_yscale('log')
plt.legend(loc=(0.05,0.1),fontsize=15)
plt.tight_layout()
#plt.savefig('example_typical_sep_cont.png')
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    Cell In[28], line 50
         46 phaseisclosetotypicalcont = np.where(np.isclose(typical_cont,orbits['Contrasts'][i],atol=1e-9))[0]
         48 plt.scatter(phaseisclosetotypicalcont,orbits['Contrasts'][i][phaseisclosetotypicalcont],
         49            marker='x',color='lightseagreen',s=100, label='typical phase')
    ---> 50 string3 = [int(orbits['Phases'][i][phaseisclosetotypicalcont][j]) for j in range(2)]
         51 yoff = [-25,-20]
         52 xoff = [10,20]


    IndexError: index 0 is out of bounds for axis 0 with size 0



    
![png](output_40_1.png)
    



```python
lod_elt = (0.2063 * 0.8 / 39) * 1000
lod_gmagaox = (0.2063 * 0.8 / 25.4) * 1000
lod_magaox = (0.2063 * 0.8 / 6.5) * 1000

###### Do all and save in db:
TypicalSeparation_lod_gmagaox = []
TypicalSeparation_mas_gmagaox = []
TypicalSeparation_au_gmagaox = []
TypicalPhase_gmagaox = []
TypicalCont_gmagaox = []

TypicalSeparation_lod_magaox = []
TypicalSeparation_mas_magaox = []
TypicalSeparation_au_magaox = []
TypicalPhase_magaox = []
TypicalCont_magaox = []

TypicalSeparation_lod_elt = []
TypicalSeparation_mas_elt = []
TypicalSeparation_au_elt = []
TypicalPhase_elt = []
TypicalCont_elt = []

IWA = 0.5

for i in range(len(orbits)):
    iwa1 = np.where(orbits['SepsInPlaneOfSky_lod_gmagaox'][i] > IWA)
    seps = orbits['SepsInPlaneOfSky_lod_gmagaox'][i][iwa1]
    conts = orbits['Contrasts'][i][iwa1]
    phases = orbits['Phases'][i][iwa1]
    typical_sep = np.sum(seps*(conts**2))/np.sum((conts**2))
    typical_cont = np.sum(conts*(conts**2))/np.sum((conts**2))
    typical_phase = np.sum(phases*(conts**2))/np.sum((conts**2))
    TypicalSeparation_lod_gmagaox.append(typical_sep)
    TypicalPhase_gmagaox.append(typical_phase)
    TypicalCont_gmagaox.append(typical_cont)
    
    typical_sep_mas = typical_sep * lod_gmagaox
    TypicalSeparation_mas_gmagaox.append(typical_sep_mas)
    typical_sep_au = (typical_sep_mas/1000) * orbits['sy_dist'][i] 
    TypicalSeparation_au_gmagaox.append(typical_sep_au)
    
    iwa2 = np.where(orbits['SepsInPlaneOfSky_lod_magaox'][i] > IWA)
    seps = orbits['SepsInPlaneOfSky_lod_magaox'][i][iwa2]
    conts = orbits['Contrasts'][i][iwa2]
    phases = orbits['Phases'][i][iwa2]
    if conts.shape[0] == 0:
        typical_sep = np.nan
        typical_cont = np.nan
        typical_phase = np.nan
        typical_sep_mas = np.nan
        typical_sep_au = np.nan
    else:
        typical_sep = np.sum(seps*(conts**2))/np.sum((conts**2))
        typical_cont = np.sum(conts*(conts**2))/np.sum((conts**2))
        typical_phase = np.sum(phases*(conts**2))/np.sum((conts**2))
        typical_sep_mas = typical_sep * lod_magaox
        typical_sep_au = (typical_sep_mas/1000) * orbits['sy_dist'][i] 
        
    TypicalSeparation_lod_magaox.append(typical_sep)
    TypicalPhase_magaox.append(typical_phase)
    TypicalCont_magaox.append(typical_cont)
    TypicalSeparation_mas_magaox.append(typical_sep_mas)
    TypicalSeparation_au_magaox.append(typical_sep_au)

    iwa3 = np.where(orbits['SepsInPlaneOfSky_lod_elt'][i] > IWA)
    seps = orbits['SepsInPlaneOfSky_lod_elt'][i][iwa3]
    conts = orbits['Contrasts'][i][iwa3]
    phases = orbits['Phases'][i][iwa3]
    if conts.shape[0] == 0:
        typical_sep = np.nan
        typical_cont = np.nan
        typical_phase = np.nan
        typical_sep_mas = np.nan
        typical_sep_au = np.nan
    else:
        typical_sep = np.sum(seps*(conts**2))/np.sum((conts**2))
        typical_cont = np.sum(conts*(conts**2))/np.sum((conts**2))
        typical_phase = np.sum(phases*(conts**2))/np.sum((conts**2))
        typical_sep_mas = typical_sep * lod_elt
        typical_sep_au = (typical_sep_mas/1000) * orbits['sy_dist'][i] 
        
    TypicalSeparation_lod_elt.append(typical_sep)
    TypicalPhase_elt.append(typical_phase)
    TypicalCont_elt.append(typical_cont)
    TypicalSeparation_mas_elt.append(typical_sep_mas)
    TypicalSeparation_au_elt.append(typical_sep_au)
    
```


```python
orbits['TypicalSeparation_lod_gmagaox'] = TypicalSeparation_lod_gmagaox
orbits['TypicalPhase_gmagaox'] = TypicalPhase_gmagaox
orbits['TypicalCont_gmagaox'] = TypicalCont_gmagaox

orbits['TypicalSeparation_lod_elt'] = TypicalSeparation_lod_elt
orbits['TypicalPhase_elt'] = TypicalPhase_elt
orbits['TypicalCont_elt'] = TypicalCont_elt

orbits['TypicalSeparation_lod_magaox'] = TypicalSeparation_lod_magaox
orbits['TypicalPhase_magaox'] = TypicalPhase_magaox
orbits['TypicalCont_magaox'] = TypicalCont_magaox

orbits['TypicalSeparation_mas_gmagaox'] = TypicalSeparation_mas_gmagaox
orbits['TypicalSeparation_au_gmagaox'] = TypicalSeparation_au_gmagaox

orbits['TypicalSeparation_mas_elt'] = TypicalSeparation_mas_elt
orbits['TypicalSeparation_au_elt'] = TypicalSeparation_au_elt

orbits['TypicalSeparation_mas_magaox'] = TypicalSeparation_mas_magaox
orbits['TypicalSeparation_au_magaox'] = TypicalSeparation_au_magaox

orbits.to_csv('Target-list-with-orbital-params.csv', index=False)
```

# Add planets to relevant lists


```python
orbits['Note'] = ''
names = np.array(orbits['pl_name'])
plandb = pd.read_csv('plandb.csv')

#try:
ind = np.array([], dtype=int)
for name in plandb['pl_name']:
    try:
        ind = np.append(ind,int(np.where(names == name)[0][0]))
    except IndexError:
        pass
orbits.loc[ind,'Note']='In Imaging Mission Database'

andes = ['Proxima Cen b', 'GJ 273 b', 'Wolf 1061 c', 'GJ 682 b', 'Ross 128 b']
ind = np.array([], dtype=int)
for name in andes:
    try:
        indtemp = int(np.where(names == name)[0][0])
        if orbits.loc[indtemp,'Note'] == '':
            orbits.loc[indtemp,'Note'] = 'In ELT-ANDES Golden Sample for Atm Characterization Bhatnagar+2026'
        else:
            orbits.loc[indtemp,'Note'] = orbits.loc[indtemp,'Note']+'; In ELT-ANDES Golden Sample for Atm Characterization Bhatnagar+2026'
    except IndexError:
        pass


venus = ['HD 20794 d', 'HD 219134 d', 'GJ 411 b', 'HD 219134 f', 'Proxima Cen d', 'Barnard e', 'Wolf 1061 c', 'GJ 15 A b',
             'Gl 725 A b', 'GJ 273 b', 'Barnard c', 'GJ 1061 d', 'Ross 128 b', 'GJ 251 b', 'Barnard b', 'GJ 625 b', 'Barnard d', 
             'L 98-59 f', 'GJ 1061 c', 'AU Mic d']

for name in venus:
    try:
        indtemp = int(np.where(names == name)[0][0])
        if orbits.loc[indtemp,'Note'] == '':
            orbits.loc[indtemp,'Note'] = 'In Kane+2026 Venus Zone'
        else:
            orbits.loc[indtemp,'Note'] = orbits.loc[indtemp,'Note']+'; In Kane+2026 Venus Zone'
    except IndexError:
        pass


######### HWO Target Stars List
# from https://zenodo.org/records/17195128
p = pd.read_csv('TSS25_list.csv')
rltl = pd.read_csv('Refl-light-target-list.csv')
in_TSS = []
for i in range(len(p.loc[np.where(p['TSS_tier'] == 1)])):
    ind = np.where(p.loc[np.where(p['TSS_tier'] == 1)].loc[i]['star_name'] == rltl['tic_id'])[0]
    if len(ind) > 0:
        for ii in ind:
            in_TSS.append(rltl.loc[ii,'pl_name'])

not_in_my_cat = []
for i in range(len(in_TSS)):
    ind = np.where(in_TSS[i] == orbits['pl_name'])[0]
    note = orbits.loc[ind,'Note'][ind]
    try:
        if np.isnan(float(note)):
            note = 'In HWO TSS25 Tier 1'
    except ValueError:
        note = '; In HWO TSS25 Tier 1'
    except TypeError:
        #print('not in cat')
        not_in_my_cat.append(in_TSS[i])
        pass
    orbits.loc[ind,'Note'] = note

orbits
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pl_name</th>
      <th>pl_orbsmax</th>
      <th>pl_orbper</th>
      <th>pl_orbincl</th>
      <th>pl_orbeccen</th>
      <th>pl_bmasse</th>
      <th>pl_bmassj</th>
      <th>st_mass</th>
      <th>sy_dist</th>
      <th>PlanetRadiuse</th>
      <th>...</th>
      <th>TypicalSeparation_mas_elt</th>
      <th>TypicalSeparation_au_elt</th>
      <th>TypicalSeparation_mas_magaox</th>
      <th>TypicalSeparation_au_magaox</th>
      <th>Note</th>
      <th>cHZ_inner</th>
      <th>cHZ_outer</th>
      <th>oHZ_inner</th>
      <th>oHZ_outer</th>
      <th>HZ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Proxima Cen d</td>
      <td>0.02881</td>
      <td>5.123380</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>0.260000</td>
      <td>0.000818</td>
      <td>0.1221</td>
      <td>1.30119</td>
      <td>0.692000</td>
      <td>...</td>
      <td>15.454116</td>
      <td>0.020109</td>
      <td>17.305274</td>
      <td>0.022517</td>
      <td>In Kane+2026 Venus Zone</td>
      <td>0.039254</td>
      <td>0.073942</td>
      <td>0.029215</td>
      <td>0.077999</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Proxima Cen b</td>
      <td>0.04848</td>
      <td>11.184650</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>1.055000</td>
      <td>0.003319</td>
      <td>0.1221</td>
      <td>1.30119</td>
      <td>1.020000</td>
      <td>...</td>
      <td>26.005690</td>
      <td>0.033838</td>
      <td>26.005690</td>
      <td>0.033838</td>
      <td>In ELT-ANDES Golden Sample for Atm Characteriz...</td>
      <td>0.037004</td>
      <td>0.073942</td>
      <td>0.029215</td>
      <td>0.077999</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barnard d</td>
      <td>0.01880</td>
      <td>2.340200</td>
      <td>NaN</td>
      <td>0.040</td>
      <td>0.263000</td>
      <td>0.000827</td>
      <td>0.1620</td>
      <td>1.82655</td>
      <td>0.694000</td>
      <td>...</td>
      <td>6.992167</td>
      <td>0.012772</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>In Kane+2026 Venus Zone</td>
      <td>0.062312</td>
      <td>0.115821</td>
      <td>0.046388</td>
      <td>0.122173</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Barnard b</td>
      <td>0.02290</td>
      <td>3.154200</td>
      <td>NaN</td>
      <td>0.030</td>
      <td>0.299000</td>
      <td>0.000941</td>
      <td>0.1620</td>
      <td>1.82655</td>
      <td>0.720000</td>
      <td>...</td>
      <td>8.754227</td>
      <td>0.015990</td>
      <td>12.838950</td>
      <td>0.023451</td>
      <td>In Kane+2026 Venus Zone</td>
      <td>0.062312</td>
      <td>0.115821</td>
      <td>0.046388</td>
      <td>0.122173</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Barnard e</td>
      <td>0.03810</td>
      <td>6.739200</td>
      <td>NaN</td>
      <td>0.040</td>
      <td>0.193000</td>
      <td>0.000607</td>
      <td>0.1620</td>
      <td>1.82655</td>
      <td>0.637000</td>
      <td>...</td>
      <td>14.322754</td>
      <td>0.026161</td>
      <td>16.764518</td>
      <td>0.030621</td>
      <td>In Kane+2026 Venus Zone</td>
      <td>0.062312</td>
      <td>0.115821</td>
      <td>0.046388</td>
      <td>0.122173</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>971</th>
      <td>HD 35843 c</td>
      <td>0.25000</td>
      <td>46.962200</td>
      <td>89.5800</td>
      <td>0.153</td>
      <td>11.320000</td>
      <td>0.035617</td>
      <td>0.9400</td>
      <td>69.60260</td>
      <td>2.540000</td>
      <td>...</td>
      <td>2.813075</td>
      <td>0.195797</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>0.796242</td>
      <td>1.459088</td>
      <td>0.651213</td>
      <td>1.538978</td>
      <td>0</td>
    </tr>
    <tr>
      <th>972</th>
      <td>HD 179079 b</td>
      <td>0.12140</td>
      <td>14.479000</td>
      <td>0.3766</td>
      <td>0.049</td>
      <td>4195.334972</td>
      <td>13.200000</td>
      <td>1.1400</td>
      <td>69.71160</td>
      <td>11.290446</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>1.438412</td>
      <td>2.637348</td>
      <td>1.176413</td>
      <td>2.781752</td>
      <td>0</td>
    </tr>
    <tr>
      <th>973</th>
      <td>HD 72892 b</td>
      <td>0.22800</td>
      <td>39.446030</td>
      <td>NaN</td>
      <td>0.419</td>
      <td>1737.885729</td>
      <td>5.468000</td>
      <td>1.0100</td>
      <td>69.73110</td>
      <td>12.800000</td>
      <td>...</td>
      <td>3.040241</td>
      <td>0.211999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>1.089960</td>
      <td>1.996081</td>
      <td>0.891436</td>
      <td>2.105372</td>
      <td>0</td>
    </tr>
    <tr>
      <th>974</th>
      <td>TOI-1648 b</td>
      <td>0.06940</td>
      <td>7.331602</td>
      <td>88.2900</td>
      <td>0.178</td>
      <td>7.400000</td>
      <td>0.023300</td>
      <td>0.8300</td>
      <td>69.83500</td>
      <td>2.540000</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>0.550020</td>
      <td>1.037373</td>
      <td>0.449797</td>
      <td>1.094189</td>
      <td>0</td>
    </tr>
    <tr>
      <th>975</th>
      <td>eps CrB b</td>
      <td>1.30000</td>
      <td>417.900000</td>
      <td>NaN</td>
      <td>0.110</td>
      <td>2129.461000</td>
      <td>6.700000</td>
      <td>1.7000</td>
      <td>69.86830</td>
      <td>12.700000</td>
      <td>...</td>
      <td>13.673755</td>
      <td>0.955362</td>
      <td>15.955882</td>
      <td>1.114810</td>
      <td></td>
      <td>11.970436</td>
      <td>23.030417</td>
      <td>9.789229</td>
      <td>24.292074</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>976 rows × 58 columns</p>
</div>




```python
not_in_my_cat
```




    ['rho CrB d', 'HN Peg b', 'tau Cet e', 'HD 102365 b', 'HD 20794 f']



tau Cet e, HD 102365 b, rho CrB d identified as false postive planet 

HN Peg b, HD 20794 f dropped b/c missing orbit info


```python
def GetHZ(StarTeff, StarRad, PlanetMass):
    ''' Get the conservative habitable zone (inner edge: runaway greenhouse, outer edge: maximum greenhouse)
    and optimistc habtiable zone (inner edge: recent Venus, outer edge: early Mars) using Eqn 4 and 5 of Kopparapu et al. 2014

    Args:
        StarTeff (flt): star effective temperature
        StarRad (flt): star radius in solar radii
        PlanetMass (flt): planet mass in Earth masses

    Returns:
        dict: inner and outer limits of consv and optimistic HZ in au
    '''
    PlanetMass = PlanetMass
    koppDict = {
        1:{'Seff':[1.776, 1.107, 0.356, 0.320],
          'a':[2.136e-4, 1.332e-4, 6.17e-5, 5.547e-5],
           'b':[2.533e-8, 1.58e-8, 1.698e-9, 1.526e-9],
           'c':[-1.332e-11, -8.308e-12, -3.198e-12,-2.874e-12],
           'd':[-3.097e-15, -1.931e-15, -5.575e-16, -5.011e-16]
          },
        5:{'Seff':[1.776, 1.188, 0.356, 0.320],
          'a':[2.136e-4, 1.433e-4, 6.17e-5, 5.547e-5],
           'b':[2.533e-8, 1.707e-8, 1.698e-9, 1.526e-9],
           'c':[-1.332e-11, -8.968e-12, -3.198e-12,-2.874e-12],
           'd':[-3.097e-15, -2.084e-15, -5.575e-16, -5.011e-16]
          },
        0.1:{'Seff':[1.776, 0.99, 0.356, 0.320],
            'a':[2.136e-4, 1.209e-4, 6.17e-5, 5.547e-5],
             'b':[2.533e-8, 1.404e-8, 1.698e-9, 1.526e-9],
             'c':[-1.332e-11, -7.418e-12, -3.198e-12,-2.874e-12],
             'd':[-3.097e-15, -1.713e-15, -5.575e-16, -5.011e-16]
            }
    }
    keys = np.array([key for key in koppDict.keys()])
    diff = np.abs(PlanetMass - keys)
    key = keys[np.where(diff == min(diff))[0]]
    sDict = koppDict[key[0]]
    T_star = StarTeff - 5780

    StarLum = (StarRad)**2 * (StarTeff/5780)**4

    outDict = {}
    keys = ['OptInner','ConsvInner','ConsvOuter','OptOuter']
    for i in range(4):
        Seff = sDict['Seff'][i] + sDict['a'][i]*T_star + sDict['b'][i]*T_star**2 + sDict['c'][i]*T_star**3 + sDict['d'][i]*T_star**4
        d = (StarLum / Seff)**(0.5)
        outDict.update({keys[i]:d})
    return outDict
        
    
```


```python
GetHZ(orbits.loc[0]['StarTeff'], orbits.loc[0]['StarRad'], orbits.loc[0]['pl_bmasse'])
```




    {'OptInner': 0.029215191583208688,
     'ConsvInner': 0.039254164923015776,
     'ConsvOuter': 0.07394227469106818,
     'OptOuter': 0.07799863680523228}




```python
orbits['cHZ_inner'] = np.nan
orbits['cHZ_outer'] = np.nan
orbits['oHZ_inner'] = np.nan
orbits['oHZ_outer'] = np.nan
orbits['HZ'] = 0
for i in range(len(orbits)):
    hzDict = GetHZ(orbits.loc[i]['StarTeff'], orbits.loc[i]['StarRad'], orbits.loc[i]['pl_bmasse'])
    orbits.loc[i,'cHZ_inner'] = hzDict['ConsvInner']
    orbits.loc[i,'cHZ_outer'] = hzDict['ConsvOuter']
    orbits.loc[i,'oHZ_inner'] = hzDict['OptInner']
    orbits.loc[i,'oHZ_outer'] = hzDict['OptOuter']

```


```python
for i in range(len(orbits)):
    if orbits.loc[i,'pl_orbsmax'] < orbits.loc[i,'cHZ_outer'] and orbits.loc[i,'pl_orbsmax'] > orbits.loc[i,'cHZ_inner'] and orbits.loc[i,'pl_bmasse'] <= 5:
        if orbits.loc[i,'Note'] == '':
            orbits.loc[i,'Note'] = 'In Conservative HZ'
        else:
            orbits.loc[i,'Note'] = orbits.loc[indtemp,'Note']+'; In Conservative HZ'
        orbits.loc[i,'HZ'] = 1
    elif orbits.loc[i,'pl_orbsmax'] < orbits.loc[i,'oHZ_outer'] and orbits.loc[i,'pl_orbsmax'] > orbits.loc[i,'oHZ_inner']  and orbits.loc[i,'pl_bmasse'] <= 5:
        if orbits.loc[i,'Note'] == '':
            orbits.loc[i,'Note'] = 'In Optimistic HZ'
        else:
            orbits.loc[i,'Note'] = orbits.loc[indtemp,'Note']+'; In Optimistic HZ'
        orbits.loc[i,'HZ'] = 2
```


```python
orbits
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pl_name</th>
      <th>pl_orbsmax</th>
      <th>pl_orbper</th>
      <th>pl_orbincl</th>
      <th>pl_orbeccen</th>
      <th>pl_bmasse</th>
      <th>pl_bmassj</th>
      <th>st_mass</th>
      <th>sy_dist</th>
      <th>PlanetRadiuse</th>
      <th>...</th>
      <th>TypicalSeparation_mas_elt</th>
      <th>TypicalSeparation_au_elt</th>
      <th>TypicalSeparation_mas_magaox</th>
      <th>TypicalSeparation_au_magaox</th>
      <th>Note</th>
      <th>cHZ_inner</th>
      <th>cHZ_outer</th>
      <th>oHZ_inner</th>
      <th>oHZ_outer</th>
      <th>HZ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Proxima Cen d</td>
      <td>0.02881</td>
      <td>5.123380</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>0.260000</td>
      <td>0.000818</td>
      <td>0.1221</td>
      <td>1.30119</td>
      <td>0.692000</td>
      <td>...</td>
      <td>15.454116</td>
      <td>0.020109</td>
      <td>17.305274</td>
      <td>0.022517</td>
      <td>In Kane+2026 Venus Zone</td>
      <td>0.039254</td>
      <td>0.073942</td>
      <td>0.029215</td>
      <td>0.077999</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Proxima Cen b</td>
      <td>0.04848</td>
      <td>11.184650</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>1.055000</td>
      <td>0.003319</td>
      <td>0.1221</td>
      <td>1.30119</td>
      <td>1.020000</td>
      <td>...</td>
      <td>26.005690</td>
      <td>0.033838</td>
      <td>26.005690</td>
      <td>0.033838</td>
      <td>In Kane+2026 Venus Zone; In Conservative HZ</td>
      <td>0.037004</td>
      <td>0.073942</td>
      <td>0.029215</td>
      <td>0.077999</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barnard d</td>
      <td>0.01880</td>
      <td>2.340200</td>
      <td>NaN</td>
      <td>0.040</td>
      <td>0.263000</td>
      <td>0.000827</td>
      <td>0.1620</td>
      <td>1.82655</td>
      <td>0.694000</td>
      <td>...</td>
      <td>6.992167</td>
      <td>0.012772</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>In Kane+2026 Venus Zone</td>
      <td>0.062312</td>
      <td>0.115821</td>
      <td>0.046388</td>
      <td>0.122173</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Barnard b</td>
      <td>0.02290</td>
      <td>3.154200</td>
      <td>NaN</td>
      <td>0.030</td>
      <td>0.299000</td>
      <td>0.000941</td>
      <td>0.1620</td>
      <td>1.82655</td>
      <td>0.720000</td>
      <td>...</td>
      <td>8.754227</td>
      <td>0.015990</td>
      <td>12.838950</td>
      <td>0.023451</td>
      <td>In Kane+2026 Venus Zone</td>
      <td>0.062312</td>
      <td>0.115821</td>
      <td>0.046388</td>
      <td>0.122173</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Barnard e</td>
      <td>0.03810</td>
      <td>6.739200</td>
      <td>NaN</td>
      <td>0.040</td>
      <td>0.193000</td>
      <td>0.000607</td>
      <td>0.1620</td>
      <td>1.82655</td>
      <td>0.637000</td>
      <td>...</td>
      <td>14.322754</td>
      <td>0.026161</td>
      <td>16.764518</td>
      <td>0.030621</td>
      <td>In Kane+2026 Venus Zone</td>
      <td>0.062312</td>
      <td>0.115821</td>
      <td>0.046388</td>
      <td>0.122173</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>971</th>
      <td>HD 35843 c</td>
      <td>0.25000</td>
      <td>46.962200</td>
      <td>89.5800</td>
      <td>0.153</td>
      <td>11.320000</td>
      <td>0.035617</td>
      <td>0.9400</td>
      <td>69.60260</td>
      <td>2.540000</td>
      <td>...</td>
      <td>2.813075</td>
      <td>0.195797</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>0.796242</td>
      <td>1.459088</td>
      <td>0.651213</td>
      <td>1.538978</td>
      <td>0</td>
    </tr>
    <tr>
      <th>972</th>
      <td>HD 179079 b</td>
      <td>0.12140</td>
      <td>14.479000</td>
      <td>0.3766</td>
      <td>0.049</td>
      <td>4195.334972</td>
      <td>13.200000</td>
      <td>1.1400</td>
      <td>69.71160</td>
      <td>11.290446</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>1.438412</td>
      <td>2.637348</td>
      <td>1.176413</td>
      <td>2.781752</td>
      <td>0</td>
    </tr>
    <tr>
      <th>973</th>
      <td>HD 72892 b</td>
      <td>0.22800</td>
      <td>39.446030</td>
      <td>NaN</td>
      <td>0.419</td>
      <td>1737.885729</td>
      <td>5.468000</td>
      <td>1.0100</td>
      <td>69.73110</td>
      <td>12.800000</td>
      <td>...</td>
      <td>3.040241</td>
      <td>0.211999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>1.089960</td>
      <td>1.996081</td>
      <td>0.891436</td>
      <td>2.105372</td>
      <td>0</td>
    </tr>
    <tr>
      <th>974</th>
      <td>TOI-1648 b</td>
      <td>0.06940</td>
      <td>7.331602</td>
      <td>88.2900</td>
      <td>0.178</td>
      <td>7.400000</td>
      <td>0.023300</td>
      <td>0.8300</td>
      <td>69.83500</td>
      <td>2.540000</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>0.550020</td>
      <td>1.037373</td>
      <td>0.449797</td>
      <td>1.094189</td>
      <td>0</td>
    </tr>
    <tr>
      <th>975</th>
      <td>eps CrB b</td>
      <td>1.30000</td>
      <td>417.900000</td>
      <td>NaN</td>
      <td>0.110</td>
      <td>2129.461000</td>
      <td>6.700000</td>
      <td>1.7000</td>
      <td>69.86830</td>
      <td>12.700000</td>
      <td>...</td>
      <td>13.673755</td>
      <td>0.955362</td>
      <td>15.955882</td>
      <td>1.114810</td>
      <td></td>
      <td>11.970436</td>
      <td>23.030417</td>
      <td>9.789229</td>
      <td>24.292074</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>976 rows × 58 columns</p>
</div>




```python
orbits.columns
```




    Index(['pl_name', 'pl_orbsmax', 'pl_orbper', 'pl_orbincl', 'pl_orbeccen',
           'pl_bmasse', 'pl_bmassj', 'st_mass', 'sy_dist', 'PlanetRadiuse',
           'pl_orblper', 'StarTeff', 'st_logg', 'st_met', 'StarRad', 'sy_imag',
           'sy_gaiamag', 'rastr', 'ra', 'decstr', 'dec', 'SpTNumber',
           'MaxProjectedSeparation_au', 'MaxProjectedSeparation_mas',
           'PhaseAtMaxProj', 'ContrastAtMaxProj', 'MaxProjectedSeparation_lod_elt',
           'MaxProjectedSeparation_lod_gmagaox',
           'MaxProjectedSeparation_lod_magaox', 'SepsInPlaneOfSky_au',
           'SepsInPlaneOfOrbit_au', 'Contrasts', 'Phases', 'SepsInPlaneOfSky_mas',
           'SepsInPlaneOfSky_lod_magaox', 'SepsInPlaneOfSky_lod_gmagaox',
           'SepsInPlaneOfSky_lod_elt', 'TypicalSeparation_lod_gmagaox',
           'TypicalPhase_gmagaox', 'TypicalCont_gmagaox',
           'TypicalSeparation_lod_elt', 'TypicalPhase_elt', 'TypicalCont_elt',
           'TypicalSeparation_lod_magaox', 'TypicalPhase_magaox',
           'TypicalCont_magaox', 'TypicalSeparation_mas_gmagaox',
           'TypicalSeparation_au_gmagaox', 'TypicalSeparation_mas_elt',
           'TypicalSeparation_au_elt', 'TypicalSeparation_mas_magaox',
           'TypicalSeparation_au_magaox', 'Note', 'cHZ_inner', 'cHZ_outer',
           'oHZ_inner', 'oHZ_outer', 'HZ'],
          dtype='object')




```python
def EarthEqIntellation(Rstar, Tstar, Sep):
    ''' Scale incident power 
    
    Args:
        Rstar (flt): Rsun
        Tsat (flt): K
        Sep (flt): AU

    Returns:
        flt: Incident power in Earth units
    '''
    return (Rstar/Sep)**2 * (Tstar/5778)**4

orbits['EarthEqInstell'] = np.nan
for i in range(len(orbits)):
    orbits.loc[i,'EarthEqInstell'] = EarthEqIntellation(orbits.loc[i,'StarRad'],
                                                       orbits.loc[i,'StarTeff'],
                                                       orbits.loc[i,'pl_orbsmax'])
```


```python
orbits
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pl_name</th>
      <th>pl_orbsmax</th>
      <th>pl_orbper</th>
      <th>pl_orbincl</th>
      <th>pl_orbeccen</th>
      <th>pl_bmasse</th>
      <th>pl_bmassj</th>
      <th>st_mass</th>
      <th>sy_dist</th>
      <th>PlanetRadiuse</th>
      <th>...</th>
      <th>TypicalSeparation_au_elt</th>
      <th>TypicalSeparation_mas_magaox</th>
      <th>TypicalSeparation_au_magaox</th>
      <th>Note</th>
      <th>cHZ_inner</th>
      <th>cHZ_outer</th>
      <th>oHZ_inner</th>
      <th>oHZ_outer</th>
      <th>HZ</th>
      <th>EarthEqInstell</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Proxima Cen d</td>
      <td>0.02881</td>
      <td>5.123380</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>0.260000</td>
      <td>0.000818</td>
      <td>0.1221</td>
      <td>1.30119</td>
      <td>0.692000</td>
      <td>...</td>
      <td>0.020109</td>
      <td>17.305274</td>
      <td>0.022517</td>
      <td>In Kane+2026 Venus Zone</td>
      <td>0.039254</td>
      <td>0.073942</td>
      <td>0.029215</td>
      <td>0.077999</td>
      <td>0</td>
      <td>1.519965</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Proxima Cen b</td>
      <td>0.04848</td>
      <td>11.184650</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>1.055000</td>
      <td>0.003319</td>
      <td>0.1221</td>
      <td>1.30119</td>
      <td>1.020000</td>
      <td>...</td>
      <td>0.033838</td>
      <td>26.005690</td>
      <td>0.033838</td>
      <td>In Kane+2026 Venus Zone; In Conservative HZ</td>
      <td>0.037004</td>
      <td>0.073942</td>
      <td>0.029215</td>
      <td>0.077999</td>
      <td>1</td>
      <td>0.536778</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barnard d</td>
      <td>0.01880</td>
      <td>2.340200</td>
      <td>NaN</td>
      <td>0.040</td>
      <td>0.263000</td>
      <td>0.000827</td>
      <td>0.1620</td>
      <td>1.82655</td>
      <td>0.694000</td>
      <td>...</td>
      <td>0.012772</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>In Kane+2026 Venus Zone</td>
      <td>0.062312</td>
      <td>0.115821</td>
      <td>0.046388</td>
      <td>0.122173</td>
      <td>0</td>
      <td>9.053178</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Barnard b</td>
      <td>0.02290</td>
      <td>3.154200</td>
      <td>NaN</td>
      <td>0.030</td>
      <td>0.299000</td>
      <td>0.000941</td>
      <td>0.1620</td>
      <td>1.82655</td>
      <td>0.720000</td>
      <td>...</td>
      <td>0.015990</td>
      <td>12.838950</td>
      <td>0.023451</td>
      <td>In Kane+2026 Venus Zone</td>
      <td>0.062312</td>
      <td>0.115821</td>
      <td>0.046388</td>
      <td>0.122173</td>
      <td>0</td>
      <td>6.101629</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Barnard e</td>
      <td>0.03810</td>
      <td>6.739200</td>
      <td>NaN</td>
      <td>0.040</td>
      <td>0.193000</td>
      <td>0.000607</td>
      <td>0.1620</td>
      <td>1.82655</td>
      <td>0.637000</td>
      <td>...</td>
      <td>0.026161</td>
      <td>16.764518</td>
      <td>0.030621</td>
      <td>In Kane+2026 Venus Zone</td>
      <td>0.062312</td>
      <td>0.115821</td>
      <td>0.046388</td>
      <td>0.122173</td>
      <td>0</td>
      <td>2.204280</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>971</th>
      <td>HD 35843 c</td>
      <td>0.25000</td>
      <td>46.962200</td>
      <td>89.5800</td>
      <td>0.153</td>
      <td>11.320000</td>
      <td>0.035617</td>
      <td>0.9400</td>
      <td>69.60260</td>
      <td>2.540000</td>
      <td>...</td>
      <td>0.195797</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>0.796242</td>
      <td>1.459088</td>
      <td>0.651213</td>
      <td>1.538978</td>
      <td>0</td>
      <td>11.904222</td>
    </tr>
    <tr>
      <th>972</th>
      <td>HD 179079 b</td>
      <td>0.12140</td>
      <td>14.479000</td>
      <td>0.3766</td>
      <td>0.049</td>
      <td>4195.334972</td>
      <td>13.200000</td>
      <td>1.1400</td>
      <td>69.71160</td>
      <td>11.290446</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>1.438412</td>
      <td>2.637348</td>
      <td>1.176413</td>
      <td>2.781752</td>
      <td>0</td>
      <td>164.358126</td>
    </tr>
    <tr>
      <th>973</th>
      <td>HD 72892 b</td>
      <td>0.22800</td>
      <td>39.446030</td>
      <td>NaN</td>
      <td>0.419</td>
      <td>1737.885729</td>
      <td>5.468000</td>
      <td>1.0100</td>
      <td>69.73110</td>
      <td>12.800000</td>
      <td>...</td>
      <td>0.211999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>1.089960</td>
      <td>1.996081</td>
      <td>0.891436</td>
      <td>2.105372</td>
      <td>0</td>
      <td>26.889218</td>
    </tr>
    <tr>
      <th>974</th>
      <td>TOI-1648 b</td>
      <td>0.06940</td>
      <td>7.331602</td>
      <td>88.2900</td>
      <td>0.178</td>
      <td>7.400000</td>
      <td>0.023300</td>
      <td>0.8300</td>
      <td>69.83500</td>
      <td>2.540000</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>0.550020</td>
      <td>1.037373</td>
      <td>0.449797</td>
      <td>1.094189</td>
      <td>0</td>
      <td>67.625097</td>
    </tr>
    <tr>
      <th>975</th>
      <td>eps CrB b</td>
      <td>1.30000</td>
      <td>417.900000</td>
      <td>NaN</td>
      <td>0.110</td>
      <td>2129.461000</td>
      <td>6.700000</td>
      <td>1.7000</td>
      <td>69.86830</td>
      <td>12.700000</td>
      <td>...</td>
      <td>0.955362</td>
      <td>15.955882</td>
      <td>1.114810</td>
      <td></td>
      <td>11.970436</td>
      <td>23.030417</td>
      <td>9.789229</td>
      <td>24.292074</td>
      <td>0</td>
      <td>88.230696</td>
    </tr>
  </tbody>
</table>
<p>976 rows × 59 columns</p>
</div>




```python
orbits.to_csv('Target-list-with-orbital-params.csv', index=False)
```

# Bokeh Interactive Plot


```python
######### This will work in  bokeh 2.4.3.


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


def get_line_eq(x0, x1, y0, y1):
    return y0 - y1, x1 - x0, x0 * y1 - x1 * y0
    
def GetPointsAboveContrastCurve(x,y,cont_curve):
    aboves = []
    for i in range(len(cont_curve[0])-1):
        ar, br, cr = get_line_eq(cont_curve[0][i], cont_curve[0][i+1], cont_curve[1][i], cont_curve[1][i+1])
        above = (ar * x + br * y + cr >= 0) 
        xaboves.append(x[above])
        yaboves.append(y[above])
        aboves.append(above)
    
    aboves = np.array(aboves)
    above = np.empty(aboves.shape[1], dtype=bool)
    for i in range(aboves.shape[1]):
        above[i] = np.all(aboves[:,i])
    return above

def MakeInteractiveSeparationContrastPlotOfNearbyRVPlanets(orbits, plotx, ploty, phases, 
                                                           saveplot = True, 
                                                           sepau = None,
                                                           sepmas = None,
                                                           filt = 'None', xaxis_label = '',
                                                           annotation_text = '', IWA = 2,
                                                           ytop = 6e-6, ybottom = 2e-10,
                                                           xright = 20, xleft = 0,
                                                           ncolors = 10, ticklocs = 'None', ticklabels = 'None',
                                                          output_file_name = 'RVPlanetContrastPlot',
                                                          cont_curve = None):


    rad = orbits['PlanetRadiuse'].copy()
    spt = orbits['SpTNumber'].copy()
    
    plotx, ploty = np.array(plotx),np.array(ploty)
    multiplier = 2
    datadf = pd.DataFrame(data={'plotx':plotx, 'ploty':ploty, 'color':spt, 'markersize':rad*multiplier,
                               'name':orbits['pl_name'], 'rad':rad, 'spt':spt, 'dist':orbits['sy_dist'],
                                'phases':phases, 'plotx_og':plotx, 'ploty_og':ploty, 'iwa': 2, 
                                'sepau':sepau, 'sepmas':sepmas, 'dec':orbits['dec'], 
                                'starteff':orbits['StarTeff'],
                                'masse':orbits['pl_bmasse']
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
        ("Sep [GmagAOX i' lod]", '@plotx{0.0}'),
        ('Sep [au]', '@sepau{0.00}'),
        ('Sep [mas]', '@sepmas{0.00}'),
        ('Rad [Rearth]','@rad{0.00}'),
        ('Mass or Msini [Mearth]','@masse{0.0}'),
        ('Star Teff', '@starteff{0}'),
        ('SpT','@spt{0.0}'),
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

    if cont_curve is None:
        pass
    else:
        gmt_lod = (0.2063 * 0.8 / 24.5) * 1000
        cont_curve[0] = [cont_curve[0][i]/gmt_lod for i in range(len(cont_curve[0]))]
        cont_curve[0].append(max(data.data['plotx']))
        cont_curve[0].append(max(data.data['plotx']))
        cont_curve[0] = [cont_curve[0][0]] + cont_curve[0]
        cont_curve[0] = cont_curve[0] + [cont_curve[0][0]]
        cont_curve[1].append(cont_curve[1][len(cont_curve[1])-1])
        cont_curve[1].append(1e-4)
        cont_curve[1] = [1e-4]+cont_curve[1]
        cont_curve[1] = cont_curve[1] + [1e-4]
        cont_curve = np.array(cont_curve).T
        
        # gmt_lod = (0.2063 * 0.8 / 24.5) * 1000
        # cont_curve[0] = [cont_curve[0][i]/gmt_lod for i in range(len(cont_curve[0]))]
        # # extend all the way out:
        # cont_curve[0].append(max(plotx))
        # cont_curve[1].append(cont_curve[1][len(cont_curve[1])-1])
        # # extend up:
        # cont_curve[0] = [cont_curve[0][0]]+cont_curve[0]
        # cont_curve[1] = [1e-4]+cont_curve[1]
        # cont_curve = np.array(cont_curve)
        
        p.line(np.array(cont_curve[:,0]),cont_curve[:,1])
        #p.line(np.array(cont_curve[0]),cont_curve[1])
    
        #points = GetPointsAboveContrastCurve(plotx,ploty,cont_curve)
    
        points = GetPointsWithinARegion(data.data['plotx'], data.data['ploty'], cont_curve)
        datadfpoints = pd.DataFrame(data={'plotx':plotx[points], 'ploty':ploty[points], 'markersize':rad[points]*multiplier,
                                          'phases':phases[points], 'color':spt[points], 
                                   'name':orbits['pl_name'][points], 'rad':rad[points], 'spt':spt[points], 'dist':orbits['sy_dist'][points],
                                    'phases':phases[points], 'plotx_og':plotx[points], 'ploty_og':ploty[points], 'iwa': 2, 
                                    'sepau':sepau[points], 'sepmas':sepmas[points], 'dec':orbits['dec'][points], 
                                    'starteff':orbits['StarTeff'][points],
                                    'masse':orbits['pl_bmasse'][points]
                                   })
        datadfpoints = datadfpoints.reset_index(drop=True)
        datadfpointsdict = datadfpoints.to_dict(orient = 'list')
        datapoints=ColumnDataSource(data=datadfpointsdict)
        p.scatter('plotx','ploty', source=datapoints, fill_alpha=1, size='markersize', 
                 line_color='black', color=None, line_width=3)

    color_bar = ColorBar(color_mapper=mapper['transform'], width=15, 
                         location=(0,0), title="Phase",
                        title_text_font_size = '20pt',
                         major_label_text_font_size = '15pt')

    p.add_layout(color_bar, 'right')

    label = Label(
        text= annotation_text,
        x=50, y=20,
        x_units="screen", y_units="screen",text_font_size = '20pt')#,render_mode="css"
    #)
    p.add_layout(label)
    
    delt = np.log10(ytop) - np.log10(ybottom)

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
        var newy = y.map(m => m * Ag/0.45 );
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


    #show(column(p, row(AgSlider),row(LambdaSlider),row(DSlider)))
    # # output_file(output_file_name+".html")
    # # save(column(p, row(AgSlider),row(LambdaSlider),row(DSlider)))
    show(p)

    return p

orbits = pd.read_csv('Target-list-with-orbital-params.csv')


ind = np.where((np.array(orbits['MaxProjectedSeparation_lod_gmagaox']) > 0.5) & 
               (np.array(orbits['dec']) < 20) & 
               (np.array(orbits['dec']) > -65) 
              )[0]


MakeInteractiveSeparationContrastPlotOfNearbyRVPlanets(orbits.loc[ind], 
                       np.array(orbits['MaxProjectedSeparation_lod_gmagaox'])[ind],
                       np.array(orbits['ContrastAtMaxProj'])[ind], 
                       np.array(orbits['PhaseAtMaxProj'])[ind],
                       sepau = np.array(orbits['MaxProjectedSeparation_au'])[ind],
                       sepmas = np.array(orbits['MaxProjectedSeparation_mas'])[ind],
                       filt = "i'",
                       xaxis_label = r'\[ \mathrm{Max\; Projected\; Separation}\; [\lambda/D]\]',
                       annotation_text = '',
                       output_file_name = 'TargetList-GMagAOX-Max-separation-contrast',
                                                      cont_curve = None)
```




<div style="display: table;"><div style="display: table-row;"><div style="display: table-cell;"><b title="bokeh.plotting._figure.figure">figure</b>(</div><div style="display: table-cell;">id&nbsp;=&nbsp;'p1007', <span id="p1078" style="cursor: pointer;">&hellip;)</span></div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">above&nbsp;=&nbsp;[],</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">align&nbsp;=&nbsp;'auto',</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">aspect_ratio&nbsp;=&nbsp;None,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">aspect_scale&nbsp;=&nbsp;1,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">attribution&nbsp;=&nbsp;[],</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">background_fill_alpha&nbsp;=&nbsp;1.0,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">background_fill_color&nbsp;=&nbsp;'#ffffff',</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">below&nbsp;=&nbsp;[LinearAxis(id='p1018', ...)],</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">border_fill_alpha&nbsp;=&nbsp;1.0,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">border_fill_color&nbsp;=&nbsp;'#ffffff',</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">center&nbsp;=&nbsp;[Grid(id='p1022', ...), Grid(id='p1027', ...), Label(id='p1063', ...), Span(id='p1064', ...)],</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">context_menu&nbsp;=&nbsp;None,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">css_classes&nbsp;=&nbsp;[],</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">css_variables&nbsp;=&nbsp;{},</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">disabled&nbsp;=&nbsp;False,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">elements&nbsp;=&nbsp;[],</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">extra_x_ranges&nbsp;=&nbsp;{},</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">extra_x_scales&nbsp;=&nbsp;{},</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">extra_y_ranges&nbsp;=&nbsp;{},</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">extra_y_scales&nbsp;=&nbsp;{},</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">flow_mode&nbsp;=&nbsp;'block',</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">frame_align&nbsp;=&nbsp;True,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">frame_height&nbsp;=&nbsp;None,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">frame_width&nbsp;=&nbsp;None,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">height&nbsp;=&nbsp;750,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">height_policy&nbsp;=&nbsp;'auto',</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">hidpi&nbsp;=&nbsp;True,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">hold_render&nbsp;=&nbsp;False,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">js_event_callbacks&nbsp;=&nbsp;{},</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">js_property_callbacks&nbsp;=&nbsp;{},</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">left&nbsp;=&nbsp;[LogAxis(id='p1023', ...)],</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">lod_factor&nbsp;=&nbsp;10,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">lod_interval&nbsp;=&nbsp;300,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">lod_threshold&nbsp;=&nbsp;2000,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">lod_timeout&nbsp;=&nbsp;500,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">margin&nbsp;=&nbsp;None,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">match_aspect&nbsp;=&nbsp;False,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">max_height&nbsp;=&nbsp;None,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">max_width&nbsp;=&nbsp;None,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border&nbsp;=&nbsp;5,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border_bottom&nbsp;=&nbsp;None,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border_left&nbsp;=&nbsp;None,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border_right&nbsp;=&nbsp;None,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border_top&nbsp;=&nbsp;None,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_height&nbsp;=&nbsp;None,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_width&nbsp;=&nbsp;None,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">name&nbsp;=&nbsp;None,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_alpha&nbsp;=&nbsp;1.0,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_cap&nbsp;=&nbsp;'butt',</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_color&nbsp;=&nbsp;'#e5e5e5',</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_dash&nbsp;=&nbsp;[],</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_dash_offset&nbsp;=&nbsp;0,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_join&nbsp;=&nbsp;'bevel',</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_width&nbsp;=&nbsp;1,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">output_backend&nbsp;=&nbsp;'canvas',</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">renderers&nbsp;=&nbsp;[GlyphRenderer(id='p1049', ...), GlyphRenderer(id='p1058', ...)],</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">reset_policy&nbsp;=&nbsp;'standard',</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">resizable&nbsp;=&nbsp;False,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">right&nbsp;=&nbsp;[ColorBar(id='p1061', ...)],</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">sizing_mode&nbsp;=&nbsp;None,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">styles&nbsp;=&nbsp;{},</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">stylesheets&nbsp;=&nbsp;[],</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">subscribed_events&nbsp;=&nbsp;PropertyValueSet(),</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">syncable&nbsp;=&nbsp;True,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">tags&nbsp;=&nbsp;[],</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">title&nbsp;=&nbsp;Title(id='p1014', ...),</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">title_location&nbsp;=&nbsp;'above',</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">toolbar&nbsp;=&nbsp;Toolbar(id='p1015', ...),</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">toolbar_inner&nbsp;=&nbsp;False,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">toolbar_location&nbsp;=&nbsp;'above',</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">toolbar_sticky&nbsp;=&nbsp;True,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">visible&nbsp;=&nbsp;True,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">width&nbsp;=&nbsp;900,</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">width_policy&nbsp;=&nbsp;'auto',</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">x_range&nbsp;=&nbsp;Range1d(id='p1065', ...),</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">x_scale&nbsp;=&nbsp;LinearScale(id='p1016', ...),</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">y_range&nbsp;=&nbsp;Range1d(id='p1066', ...),</div></div><div class="p1077" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">y_scale&nbsp;=&nbsp;LogScale(id='p1017', ...))</div></div></div>
<script>
(function() {
  let expanded = false;
  const ellipsis = document.getElementById("p1078");
  ellipsis.addEventListener("click", function() {
    const rows = document.getElementsByClassName("p1077");
    for (let i = 0; i < rows.length; i++) {
      const el = rows[i];
      el.style.display = expanded ? "none" : "table-row";
    }
    ellipsis.innerHTML = expanded ? "&hellip;)" : "&lsaquo;&lsaquo;&lsaquo;";
    expanded = !expanded;
  });
})();
</script>





```python

```
