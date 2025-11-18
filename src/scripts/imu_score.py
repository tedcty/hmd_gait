import pandas as pd
from pathlib import Path

def calculate_sensor_importance(input_file):
    """
    Read feature importance file, group by sensor location, and sum importances.
    """
    # Read the input file
    df = pd.read_csv(input_file)
    
    # Extract sensor location (everything before first underscore)
    df['sensor_location'] = df['feature'].str.split('_', n=1).str[0]
    
    # Group by sensor location and sum importance scores
    sensor_importance = df.groupby('sensor_location')['importance'].sum().reset_index()
    sensor_importance.columns = ['sensor_location', 'importance_sum']
    
    # Save to same directory as input file
    input_path = Path(input_file)
    output_file = input_path.parent / f"{input_path.stem}_sensor_summary.csv"
    sensor_importance.to_csv(output_file, index=False)
    
    print(f"Sensor importance summary saved to: {output_file}")
    return sensor_importance

if __name__ == "__main__":
    
    input_file = "Z:/Upper Body/Results/30 Participants/models/IMU/Step_over_cone/top_100/IMU_Step over cone_feature_importances_top100.csv"
    result = calculate_sensor_importance(input_file)
    print("\nSensor Importance Summary:")
    print(result)