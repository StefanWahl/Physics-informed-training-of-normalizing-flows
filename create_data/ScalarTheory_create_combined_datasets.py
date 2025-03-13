import os
import argparse
import fnmatch
import json
import torch
import datetime

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source',                 type = str, required=True,  help = "Location of the simulation results")
    parser.add_argument('--destination',            type = str, required=True,  help = "Destination of the results")
    parser.add_argument('--n_correlation_times',    type = int, required=False, help = "Number of correlation times inbetween samples", default=2)

    args = parser.parse_args()

    source = args.source
    destination = args.destination

    n_correlation_times = args.n_correlation_times

    # Get all the availlable folders
    source_entries = os.listdir(source)

    for entry_i in source_entries:

        base_folder_i = os.path.join(source,entry_i)

        # Save the data to the configuration
        destination_folder_i = os.path.join(destination,entry_i)

        if not os.path.exists(destination_folder_i):
            os.makedirs(destination_folder_i)

        # Check if the entry is a folder
        if os.path.isdir(base_folder_i):

            # Get the content of the folder
            content_i = os.listdir(base_folder_i)
            
            # Get all the information files in the folder
            info_files = fnmatch.filter(content_i,"info_?.json")

            curated_states = None

            # Loop over the individual runs in the folder
            for info_file_ij in info_files:

                counter_ij = int(info_file_ij.split('.')[0].split("_")[-1])

                # Check if all the other required files are in the folder
                assert(f"states_{counter_ij}.pt" in content_i)

                # Load the information file
                info_path_ij = os.path.join(base_folder_i,info_file_ij)
    
                with open(info_path_ij,"r") as file:
                    info_ij = json.load(file)
                file.close()

                # Load the states
                data_path_ij = os.path.join(base_folder_i,f"states_{counter_ij}.pt")
                data_ij = torch.load(data_path_ij).detach()
                print(f"\t{len(data_ij)} instances loaded")

                # Get the part of the stored states that is in equillibrium
                lower_lim_ij = int(info_ij["t_eq"] / info_ij["freq_save_samples"])+1
                data_ij = data_ij[lower_lim_ij:]
                print(f"\t{len(data_ij)} instances in equilibrium")

                # Select states that are at least two correlation times away from each other
                step_size_ij = int(n_correlation_times * abs(info_ij["tau_action"]) / info_ij["freq_save_samples"])+1
                data_ij = data_ij[::step_size_ij]
                print(f"\t{len(data_ij)} independen instances\n")

                # Add the states to the data set
                if curated_states is None:
                    curated_states = data_ij

                else:
                    curated_states = torch.cat((curated_states,data_ij))

                # Add some more information to the info file
                info_ij["processing_meta_data"] = {}
                    
                info_ij["processing_meta_data"]["n_selected_samples"] = len(data_ij)
                info_ij["processing_meta_data"]["n_correlation_times"] = n_correlation_times
                info_ij["processing_meta_data"]["createion_date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                info_ij["processing_meta_data"]["source_file_data"] = data_path_ij

                # Save the information of the run
                with open(os.path.join(destination_folder_i,info_file_ij),"w") as temp_file:
                    json.dump(info_ij,temp_file)
                temp_file.close()

            destination_file_i = os.path.join(destination_folder_i,"states_curated.pt")

            if os.path.exists(destination_file_i):
                raise ValueError("The specified folder does already exist!")
            
            torch.save(curated_states,destination_file_i)
