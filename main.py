import pip
import pvlib
from pvlib.location import Location
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import math


A = pvlib.location.Location(53.181, 44.049,
                            tz='Europe/Saratov',
                            altitude = 30,
                            name='Penza')
print (A)

# Получение диапазона дат
times = pd.date_range(start=datetime.datetime(2015,1,1), end=datetime.datetime(2015,12,31,23), freq='1H')

# Преобразование times к местному времени
times_loc = times.tz_localize(A.pytz)

# Расчёт положения Солнца для данных условий
SPA = pvlib.solarposition.spa_python(times_loc, A.latitude, A.longitude)

# Расчёт заатмосферной солнечной радиации за заданный период времени методом Spencer
DNI_extra = pvlib.irradiance.get_extra_radiation(times_loc, method='spencer')

# Определение солнечного излучения при ясном небе
apparent_elevation = SPA['apparent_elevation']
apparent_zenith = SPA['apparent_zenith']
zenith = SPA['zenith']
airmass_relative = pvlib.atmosphere.get_relative_airmass(zenith, model='simple')
airmass_absolute = pvlib.atmosphere.get_absolute_airmass(airmass_relative, pressure=101325.0)
                                                        # time, latitude, longitude, filepath=None, interp_turbidity=True
linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(times_loc,  A.latitude, A.longitude, filepath=None, interp_turbidity=True)

SimpleClearSky = pvlib.clearsky.ineichen(apparent_zenith, airmass_absolute, linke_turbidity, altitude=0, dni_extra=DNI_extra, perez_enhancement=False)

ClearSky = SimpleClearSky

# Определение солнечной радиации при ясном небе
VerySimpleClearSky = pvlib.clearsky.simplified_solis(apparent_elevation, aod700=0.1, precipitable_water=1.0, pressure=101325.0, dni_extra=DNI_extra)


class PVGIS(object):
    """docstring"""

    def __init__(self, latitude, longitude):
        """Constructor"""
        self.latitude = latitude;
        self.longitude = longitude;
        print(PVGIS);

    def Get_data(self, B, azimuth):
        data, inputs, metadata = pvlib.iotools.get_pvgis_hourly(
            self.latitude, self.longitude,
            start=2015, end=2015,
            raddatabase='PVGIS-ERA5',
            components=False,
            surface_tilt=B, surface_azimuth=azimuth - 180,
            outputformat='json', usehorizon=True, userhorizon=None,
            pvcalculation=False, peakpower=None,
            pvtechchoice='crystSi', mountingplace='free',
            loss=0, trackingtype=0,
            optimal_surface_tilt=False, optimalangles=False,
            url='https://re.jrc.ec.europa.eu/api/v5_2/',
            map_variables=True, timeout=30);

        return data;


class LOCAL_M(object):
    """docstring"""

    def __init__(self, latitude, longitude, altitude):
        """Constructor"""
        self.latitude = latitude;
        self.longitude = longitude;
        self.altitude = altitude;
        self.times = pd.date_range(start=datetime.datetime(2015, 1, 1), end=datetime.datetime(2015, 12, 31, 23),
                                   freq='1H');
        self.times_loc = self.times.tz_localize('Europe/Saratov');

        self.SPA = pvlib.solarposition.spa_python(self.times_loc, self.latitude, self.longitude, self.altitude);

        self.DNI_extra = pvlib.irradiance.get_extra_radiation(self.times_loc, method='spencer');

        self.airmass_relative = pvlib.atmosphere.get_relative_airmass(self.SPA['apparent_zenith']);
        self.airmass_absolute = pvlib.atmosphere.get_absolute_airmass(self.airmass_relative, pressure=101325.0);
        self.linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(self.times_loc,
                                                                     latitude, longitude,
                                                                     filepath=None, interp_turbidity=True);
        self.ClearSky = pvlib.clearsky.ineichen(self.SPA['apparent_zenith'], self.airmass_absolute,
                                                self.linke_turbidity, altitude=self.altitude,
                                                dni_extra=self.DNI_extra, perez_enhancement=False);
        print(LOCAL_M);

    def Get_AOI(self, B, azimuth):
        return pvlib.irradiance.aoi(B, surface_azimuth_, self.SPA['apparent_zenith'], self.SPA['azimuth']);

    def Get_data(self, B, azimuth):
        DiffuseSky = pvlib.irradiance.perez(B, surface_azimuth_, self.ClearSky['dhi'], self.ClearSky['dni'],
                                            self.DNI_extra,
                                            self.SPA['apparent_zenith'], self.SPA['azimuth'],
                                            self.airmass_relative, model='allsitescomposite1990');
        DiffuseGround = pvlib.irradiance.get_ground_diffuse(B, self.ClearSky['ghi'], albedo=0.25, surface_type=None);

        return pvlib.irradiance.poa_components(self.Get_AOI(B, azimuth), self.ClearSky['dni'], DiffuseSky,
                                               DiffuseGround);


surface_azimuth_ = 180;

k = dict();
name = ['PVGIS', 'NPC', 'LOCAL'];
for i in name:
    k[i] = [];
print(k);

BD_interf = dict();
BD_interf['PVGIS'] = PVGIS(A.latitude, A.longitude);
BD_interf['LOCAL'] = LOCAL_M(A.latitude, A.longitude, A.altitude);

# NPC
DirectHI = pd.read_csv('C:\\Users\Пк\Desktop\\Conda\\DirectHorizontalIrradiation.csv', sep=';');
DiffuseHI = pd.read_csv('C:\\Users\Пк\Desktop\\Conda\\DiffuseHorizontalIrradiation.csv', sep=';');
NPC = pd.DataFrame(index=range(8760), columns=['HOY', 'DOY',
                                               'Month', 'Day',
                                               'Hour', 'DHI', 'DirectHI', 'Albedo']);
NPC.sample(5);

i = 0;
data_day_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
for m in range(12):
    for d in range(data_day_month[m]):
        for h in range(24):
            NPC['DHI'][i] = DiffuseHI[str(m + 1)][h] * 1000.0;
            NPC['DirectHI'][i] = DirectHI[str(m + 1)][h] * 1000.0;
            i += 1

for B in range(0, 91, 5):
    # NPC
    DiffuseGround = pvlib.irradiance.get_ground_diffuse(B, NPC['DirectHI'], 0.25, surface_type=None);
    G = pvlib.irradiance.poa_components(BD_interf['LOCAL'].Get_AOI(B, surface_azimuth_),
                                        NPC['DirectHI'],
                                        NPC['DHI'], DiffuseGround);
    k['NPC'].append([B, G['poa_global'].sum() / 1000.0]);

    # PVGIS
    k['PVGIS'].append([B, BD_interf['PVGIS'].Get_data(B, surface_azimuth_)['poa_global'].sum() / 1000.0]);

    # LOCALE
    k['LOCAL'].append([B, BD_interf['LOCAL'].Get_data(B, surface_azimuth_)['poa_global'].sum() / 1000.0]);

    # расчёт оптимального угла
Optimal_angle = {'PVGIS': [0.0, 0.0], 'NPC': [0.0, 0.0], 'LOCAL': [0.0, 0.0]};
for T in k:
    for J, K in k[T]:
        if (Optimal_angle[T][0] < K):
            Optimal_angle[T][0] = K;
            Optimal_angle[T][1] = J;
        print(K);
print(Optimal_angle);

# Задание ориентации поверхности

surf_tilt = 40
surf_az = 180

# Определение отражённой от земли составляющей, попадающей на поверхность заданной ориентации

DiffuseGround = pvlib.irradiance.get_ground_diffuse(surf_tilt, ClearSky['ghi'], albedo=0.25, surface_type=None)

# Определение угла падения солнечных лучей на поверхность солнечных модулей

AOI = pvlib.irradiance.aoi(surf_tilt, surf_az, SPA['apparent_zenith'], SPA['azimuth'])


# Определение диффузной составляющей, попадающей на поверхность заданной ориентации

AM = pvlib.atmosphere.get_relative_airmass(SPA['apparent_zenith'])

DiffuseSky_Perez = pvlib.irradiance.perez(surf_tilt, surf_az,
                                          ClearSky['dhi'], ClearSky['dni'], DNI_extra,
                                        SPA['apparent_zenith'], SPA['azimuth'],
                                        AM, model='allsitescomposite1990')

DiffuseSky = DiffuseSky_Perez

# Определение отражённой от земли составляющей, попадающей на поверхность заданной ориентации

DiffuseGround = pvlib.irradiance.get_ground_diffuse(surf_tilt, ClearSky['ghi'], albedo=0.25, surface_type=None)


# Определение суммарной солнечной радиации, падающей на поверхность солнечных модулей

G = pvlib.irradiance.poa_components(AOI, ClearSky['dni'], DiffuseSky, DiffuseGround)
# Получение информации о модулях

PV_modules = pvlib.pvsystem.retrieve_sam(path='C:\\Users\Пк\Desktop\\Conda\\CEC Modules.csv')

PV_modules.head()
# Информация о конкретном модуле №1

PV_module1 = PV_modules['Ablytek_6MN6A290']
PV_module1

# Информация о конкретном модуле №2

PV_module2 = PV_modules['Zytech_Solar_ZT320P']
PV_module2

# Информация о конкретном модуле №3

PV_module3 = PV_modules['Heliene_96M400']
PV_module3
mod =[PV_module1, PV_module2, PV_module3]

# Получение информации об инверторах

Inverters_CEC = pvlib.pvsystem.retrieve_sam(name=None, path='C:\\Users\Пк\Desktop\\Conda\\CEC Inverters.csv')
Inverters_CEC
Inverters_CEC2 = Inverters_CEC.transpose()
Inverters_CEC2 = Inverters_CEC2.rename_axis('Name').reset_index()
for i in range(1398):
    if Inverters_CEC2['Paco'].iloc[i] > 4000 and Inverters_CEC2['Paco'].iloc[i] < 15000:
        if Inverters_CEC2['Vac'].iloc[i] > 219 and Inverters_CEC2['Vac'].iloc[i] < 241:
            if Inverters_CEC2['Idcmax'].iloc[i] > 19:
                print(Inverters_CEC2['Name'].iloc[i])

Inverter1 = Inverters_CEC['Delta_Electronics__E8_TL_US__240V_']
Inverter1

Inverters_CEC2 = Inverters_CEC.transpose()
Inverters_CEC2 = Inverters_CEC2.rename_axis('Name').reset_index()
for i in range(1398):
    if Inverters_CEC2['Paco'].iloc[i] > 15000 and Inverters_CEC2['Paco'].iloc[i] < 1000000:
        if Inverters_CEC2['Vac'].iloc[i] > 219 and Inverters_CEC2['Vac'].iloc[i] < 241:
            if Inverters_CEC2['Idcmax'].iloc[i] > 19:
                print(Inverters_CEC2['Name'].iloc[i])
Inverter2 = Inverters_CEC['INGETEAM_POWER_TECHNOLOGY_S_A___INGECON_SUN_610TL_U_B220__220V_']
Inverter2
inv = [Inverter1, Inverter2]
#pip install NREL-PySAM
from PySAM.PySSC import PySSC


class Module_model(object):
    """docstring"""

    def __init__(self, Module):
        """Constructor"""

        self.alpha_sc = Module['alpha_sc'];
        self.beta_oc = Module['beta_oc'];

        self.a_ref = Module['a_ref'];
        self.I_L_ref = Module['I_L_ref'];
        self.I_o_ref = Module['I_o_ref'];
        self.R_sh_ref = Module['R_sh_ref'];
        self.R_s = Module['R_s'];

        self.Length = Module['Length'];
        self.Width = Module['Width'];

        self.V_oc_ref = Module['V_oc_ref'];
        self.I_sc_ref = Module['I_sc_ref'];

        print(Module_model);

    def Get_F(self):
        return self.Length * self.Width;

    def Get_data_operate(self, effective_irradiance, Cells_temperature):
        photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = pvlib.pvsystem.calcparams_desoto(
            effective_irradiance,
            Cells_temperature,
            self.alpha_sc,
            self.a_ref,
            self.I_L_ref,
            self.I_o_ref,
            self.R_sh_ref,
            self.R_s,
            EgRef=1.121,
            dEgdT=-0.0002677,
            irrad_ref=1000,
            temp_ref=25)

        VOC_max = self.V_oc_ref * (1 + self.beta_oc * (min(Cells_temperature) - 25) / 100)
        ISC_max = self.I_sc_ref * (1 + self.alpha_sc * (max(Cells_temperature) - 25) / 100)

        return pvlib.pvsystem.singlediode(
            photocurrent,
            saturation_current,
            resistance_series,
            resistance_shunt,
            nNsVth,
            ivcurve_pnts=None,
            method='newton'), VOC_max, ISC_max;


#import matplotlib.pyplot as plt
#import math


PV_Modules = pvlib.pvsystem.retrieve_sam(name=None, path='C:\\Users\Пк\Desktop\\Conda\\CEC Modules.csv');
Inverters_CEC = pvlib.pvsystem.retrieve_sam(name=None, path='C:\\Users\Пк\Desktop\\Conda\\CEC Inverters.csv')

# для поиска
PV_Modules_ = PV_Modules;
PV_Modules_ = PV_Modules_.transpose();
PV_Modules_ = PV_Modules_.rename_axis('Name').reset_index();
PV_Modules_ = PV_Modules_.to_numpy();

Inverters_CEC_ = Inverters_CEC;
Inverters_CEC_ = Inverters_CEC_.transpose();
Inverters_CEC_ = Inverters_CEC_.rename_axis('Name').reset_index();
Inverters_CEC_ = Inverters_CEC_.to_numpy();


# выбор модуля
def find_module(PV_Modules_, Idcmax, P_min, P_max):
    for Module in PV_Modules_:
        if Module[4] > P_min and Module[4] < P_max:  # STC
            if Module[2] == 'Mono-c-Si':  # Technology
                if Module[3] == 0:  # одностороний
                    if Module[7] == Module[7] and Module[8] == Module[8]:  # l & w
                        if Module[10] > Idcmax:  # Idcmax
                            return Module[0];


# выбор инвертора
def find_invertor(Inverters_CEC_, Idcmax, P_min, P_max):
    for Inverter in Inverters_CEC_:
        if Inverter[3] > P_min and Inverter[3] < P_max:  # Paco
            if Inverter[1] > 219 and Inverter[1] < 241:  # Vас
                if Inverter[12] > Idcmax:  # Idcmax
                    return Inverter[0];


def Get_Temperature_cells(data, data_meteo):
    a = -3.47;
    b = -0.0594;
    deltaT = 3;
    return pvlib.temperature.sapm_cell(data['poa_global'].values,
                                       data_meteo['temp_air'].values,
                                       data_meteo['wind_speed'].values,
                                       a, b, deltaT, irrad_ref=1000.0);


db = dict();
name = ['PVGIS', 'LOCAL'];
for i in name:
    db[i] = [];

print(db);
# радиация | температура

data_pvgis = BD_interf['PVGIS'].Get_data(Optimal_angle['PVGIS'][1], surface_azimuth_);
db['PVGIS'] = [data_pvgis['poa_global'] * 0.97, Get_Temperature_cells(data_pvgis, data_pvgis)];

data_local = BD_interf['LOCAL'].Get_data(Optimal_angle['LOCAL'][1], surface_azimuth_);
db['LOCAL'] = [data_local['poa_global'] * 0.97, Get_Temperature_cells(data_local, data_pvgis)];

mod_s = dict();
inv_s = dict();

name_mod = [find_module(PV_Modules_, 0, 290, 1000),
            find_module(PV_Modules_, 0, 310, 500),
            find_module(PV_Modules_, 0, 400, 1000)];
name_inv = [find_invertor(Inverters_CEC_, 19, 15000, 1000000),
            find_invertor(Inverters_CEC_, 19, 7000, 7800)];

for m in name_mod:
    mod_s[m] = Module_model(PV_Modules[m]);
    print(PV_Modules[m]);

for i in name_inv:
    inv_s[i] = Inverters_CEC[i];
    print(Inverters_CEC[i]);


def calculate_DC(Module, Inverter, db, F_plate):
    data, VOC_max, ISC_max = Module.Get_data_operate(db[0], db[1]);

    print(VOC_max, "|", ISC_max);
    M = math.floor(Inverter['Vdcmax'] / VOC_max)
    N = math.floor(Inverter['Idcmax'] / ISC_max)
    print("M: ", M, "| N: ", N);

    VDC_array = data['v_mp'] * M;
    PDC_array = data['p_mp'] * M * N;

    ACout = pvlib.inverter.sandia(VDC_array, PDC_array, Inverter);

    # PDC_array.plot()

    N_inv = math.floor(F_plate / (Module.Get_F() * M * N));
    N_mod = N_inv * M * N;

    print(N_inv, "|", N_mod, "|", ACout.sum() * N_inv / 1000000);  # МВТ*ч
    return ACout * N_inv / 1000000;  # МВт


# анализ вариантов
F = 40000  # м^2

for Name_bd in db:
    i = 1;
    for NN in inv_s:
        P_var = [];
        for N in mod_s:
            print(Name_bd, "|", N, "|", NN);
            P_var.append([calculate_DC(mod_s[N], inv_s[NN], db[Name_bd], F * 2 / 3), i]);
            i = i + 1;
        P_var.sort(key=lambda x: x[0].sum(), reverse=True);
        for P, N in P_var:
            P.plot(label=N);
        plt.grid();
        plt.legend();
        plt.ylabel('P, МВт');
        plt.xlabel('Время');

        plt.show();