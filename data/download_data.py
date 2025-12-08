#!/usr/bin/env python3
# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: download_data.py
# Descripción: Descarga y preprocesa datos OPSD de consumo eléctrico alemán
#              Genera 36 meses de datos (2015-2017) con división train/val/test
# ============================================================================

import pandas as pd
import numpy as np
import os
from datetime import datetime
import requests

class DataDownloader:
    """Descarga y preprocesa datos de consumo eléctrico alemán de OPSD."""
    
    def __init__(self, output_dir='data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def download_opsd_data(self):
        """
        Intenta descargar datos reales de OPSD.
        Si falla, genera datos sintéticos realistas.
        """
        print("=" * 80)
        print("📥 DESCARGANDO DATOS DE OPSD (Open Power System Data)")
        print("=" * 80)
        
        url = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
        
        try:
            print(f"\n🌐 Intentando descargar desde: {url}")
            print("   (Esto puede tardar 1-2 minutos...)")
            
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            temp_file = os.path.join(self.output_dir, 'opsd_raw.csv')
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            print("✅ Descarga exitosa!")
            
            df = pd.read_csv(temp_file, parse_dates=['utc_timestamp'])
            
            if 'DE_load_actual_entsoe_transparency' in df.columns:
                df_germany = df[['utc_timestamp', 'DE_load_actual_entsoe_transparency']].copy()
                df_germany.columns = ['timestamp', 'load_mw']
                df_germany = df_germany.dropna()
                
                df_germany['load_gwh'] = df_germany['load_mw'] / 1000
                
                print(f"✅ Datos de Alemania extraídos: {len(df_germany)} registros horarios")
                
                os.remove(temp_file)
                
                return df_germany
            else:
                print("⚠️  Columna de Alemania no encontrada. Generando datos sintéticos...")
                return self.generate_synthetic_data()
                
        except Exception as e:
            print(f"⚠️  Error al descargar datos: {e}")
            print("📊 Generando datos sintéticos realistas como alternativa...")
            return self.generate_synthetic_data()
    
    def generate_synthetic_data(self):
        """
        Genera datos sintéticos realistas.
        """
        print("\n📊 Generando datos sintéticos de consumo eléctrico alemán...")
        
        date_range = pd.date_range(start='2015-01-01',
                                   end='2017-12-31 23:00:00',
                                   freq='H')
        
        n_hours = len(date_range)
        t = np.arange(n_hours)
        
        trend = 50 + 0.0001 * t
        annual_seasonality = 10 * np.sin(2 * np.pi * t / (365.25 * 24) + np.pi/2)
        weekly_seasonality = 5 * np.sin(2 * np.pi * t / (7 * 24))
        daily_seasonality = 8 * np.sin(2 * np.pi * t / 24 + np.pi/2)
        
        np.random.seed(42)
        noise = np.random.normal(0, 2, n_hours)
        
        load_gwh = trend + annual_seasonality + weekly_seasonality + daily_seasonality + noise
        load_gwh = np.clip(load_gwh, 35, 75)
        
        df = pd.DataFrame({
            'timestamp': date_range,
            'load_gwh': load_gwh
        })
        
        print(f"✅ {len(df)} registros horarios sintéticos generados")
        return df
    
    def convert_to_monthly(self, df):
        print("\n🔄 Convirtiendo datos horarios a mensuales...")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        df_monthly = df.resample('MS').sum()
        
        df_monthly = df_monthly.loc['2015-01':'2017-12']
        
        print(f"✅ {len(df_monthly)} meses de datos generados")
        print(f"   Período: {df_monthly.index[0]} a {df_monthly.index[-1]}")
        
        return df_monthly
    
    def split_train_val_test(self, df):

        if len(df) != 36:
            raise ValueError(f"❌ Se esperaban 36 meses, pero llegaron {len(df)}")

        train = df.iloc[:30]
        val = df.iloc[30:33]
        test = df.iloc[33:36]

        print("\n✂️  División completada:")
        print(f"   📚 Train:      {len(train)} meses")
        print(f"   🔍 Validation: {len(val)} meses")
        print(f"   🧪 Test:       {len(test)} meses")

        return train, val, test

    def save_data(self, df_monthly, train, val, test):
        print("\n💾 Guardando archivos CSV...")
        
        main_file = os.path.join(self.output_dir, 'germany_monthly_power.csv')
        df_monthly.to_csv(main_file)
        print(f"   ✅ {main_file}")
        
        train.to_csv(os.path.join(self.output_dir, 'train.csv'))
        val.to_csv(os.path.join(self.output_dir, 'validation.csv'))
        test.to_csv(os.path.join(self.output_dir, 'test.csv'))
        
        print("🎉 Archivos guardados!")
    
    def run(self):
        print("\n" + "=" * 80)
        print("🚀 INICIANDO PIPELINE DE DATOS")
        print("=" * 80)
        
        df_hourly = self.download_opsd_data()
        df_monthly = self.convert_to_monthly(df_hourly)
        train, val, test = self.split_train_val_test(df_monthly)
        self.save_data(df_monthly, train, val, test)
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 80)
        print("\n📊 Archivos generados:")
        print("   - germany_monthly_power.csv  (36 meses)")
        print("   - train.csv                  (30 meses)")
        print("   - validation.csv             (3 meses)")
        print("   - test.csv                   (3 meses)")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    downloader = DataDownloader()
    downloader.run()
