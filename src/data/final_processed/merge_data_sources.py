import pandas as pd

class final_processing():
    def merge_flights_weather(self, flights_df, weather_df):
        # 1. Clean Weather
        weather_df = weather_df.copy()
        weather_df['valid'] = pd.to_datetime(weather_df['valid'], errors='coerce')
        weather_df = weather_df.dropna(subset=['valid']) # TODO: Should be done in validation & cleaning
        
        # Rename
        weather_df = weather_df.rename(columns={'airport': 'origin_airport'})
        weather_df['origin_airport'] = weather_df['origin_airport'].astype(str)
        
        # SORT IS MANDATORY
        weather_df = weather_df.sort_values('valid')

        # 2. Prepare Flights
        flights_df = flights_df.copy()
        flights_df['date_dt'] = pd.to_datetime(flights_df['date_mmddyyyy'], unit='ms')
        
        flights_df['scheduled_departure_dt'] = pd.to_datetime(
            flights_df['date_dt'].dt.strftime('%Y-%m-%d') + ' ' + flights_df['scheduled_departure_time'],
            errors='coerce'
        )

        # Clean flights
        flights_df = flights_df.dropna(subset=['scheduled_departure_dt', 'origin_airport']) # TODO: Should be done in validation & cleaning
        flights_df['origin_airport'] = flights_df['origin_airport'].astype(str)
        
        # SORT IS MANDATORY
        flights_df = flights_df.sort_values('scheduled_departure_dt')

        # 3. Merge
        merged_df = pd.merge_asof(
            flights_df,
            weather_df,
            left_on='scheduled_departure_dt',
            right_on='valid',
            by='origin_airport',
            direction='backward'
        )

        return merged_df

    def merge_airline_metadata(self, df, airlines_df):
        """Joins carrier names, types, and hub information."""
        
        df['carrier_code'] = df['carrier_code'].astype(str).str.strip()
        airlines_df['carrier_code'] = airlines_df['carrier_code'].astype(str).str.strip()
        
        merged = pd.merge(df, airlines_df, on='carrier_code', how='left')
                    
        return merged

    def merge_airport_metadata(self, df, airports_df):
        
        df['origin_airport'] = df['origin_airport'].astype(str).str.strip()
        airports_df['airport_code'] = airports_df['airport_code'].astype(str).str.strip()
        
        merged = pd.merge(
            df, 
            airports_df, 
            left_on='origin_airport', 
            right_on='airport_code', 
            how='left'
        )
        
        if 'airport_code' in merged.columns:
            merged = merged.drop(columns=['airport_code'])
            
        return merged    

    def merge_all(self):
        flights_df = pd.read_parquet("data/raw/flights/flights_combined.parquet")
        weather_df = pd.read_csv("data/interim/weather/weather_combined.csv")
        airline_df = pd.read_parquet("data/interim/airlines.parquet")
        airports_df = pd.read_parquet("data/interim/airports.parquet")
        
        final_df = self.merge_airport_metadata(
            self.merge_airline_metadata(
                self.merge_flights_weather(flights_df, weather_df), 
                airline_df
            ), 
            airports_df
        )

        final_df.to_parquet("data/processed/merged_dataset.parquet")


# Should be removed
if __name__ == "__main__":
    processor = final_processing()
    processor.merge_all()