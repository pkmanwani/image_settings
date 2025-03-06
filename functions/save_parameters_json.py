from image_fit import image_fit
import os
import json

def save_parameters_to_json(folder_path):
    results = []

    # Loop through each file in the specified directory
    for filename in os.listdir(folder_path):
        if filename.endswith('.h5'):
            file_path = os.path.join(folder_path, filename)
            # Call the image_fit function
            parameters = image_fit(file_path, roi=True)  # Assuming this returns a tuple

            # Unpack the parameters if they are returned as a tuple
            Cx_final, Cy_final, Sx_final, Sy_final, Sxy_final, Rx_final, Ry_final, Rxy_final,angle, res = parameters

            # Create a result dictionary
            result = {
                'filename': filename,
                'Cx_final': Cx_final,
                'Cy_final': Cy_final,
                'Sx_final': Sx_final,
                'Sy_final': Sy_final,
                'Sxy_final': Sxy_final,
                'Rx_final': Rx_final,
                'Ry_final': Ry_final,
                'Rxy_final': Rxy_final,
                'angle' : angle,
                'res' : res,
            }
            results.append(result)

    # Define the JSON file name based on the folder name
    json_file_path = os.path.join(folder_path, 'parameters.json')

    # Save the results to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f'Saved parameters to {json_file_path}')

# Example usage
folder_path = '2025_02_20/ThreeScreen2'  # Change this to your folder path
save_parameters_to_json(folder_path)
