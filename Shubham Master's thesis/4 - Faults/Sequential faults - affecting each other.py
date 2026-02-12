import torch
import numpy as np
import copy
import matplotlib.pyplot as plt

# Import methods from pyvista_new.py
from pyvista_new import Gempy

class GempyFaultModel(Gempy):
    """
    New class that inherits Gempy class from 4D GemPy file
    Extension of the Gempy class to handle structural discontinuities (faults)

    """

    def __init__(self, project_name, extent, resolution):

        ## calling initialization of the parent 'Gempy' class
        super().__init__(project_name, extent, resolution)
        
        # Storage for multiple faults
        # Format: {'fault_name': internal_state_dict}
        self.fault_states = {}

        # Storage for displacements
        # Format: {'fault_name': tensor([dx, dy, dz, dt])}
        self.fault_displacement_vectors = {}
        
        # Storage for structural kriging system
        self.structure_state = None

        # Default coordinate transformation function (step function)
        self.transform_function = self.default_heaviside_drift
        

    def add_fault_displacement(self, fault_name, displacement_vector):
        """
        Register displacement vector for a specific fault
        """

        self.fault_displacement_vectors[fault_name] = displacement_vector


    def set_custom_transformation(self, func):

        # function to set a custom transformation function (step, sigmoid, etc)
        self.transform_function = func
        # print("Custom transformation function set")

    def default_heaviside_drift(self, scalar_field, grid_coords, threshold, displacement_vector):
        """
        Calculates drift for a SINGLE fault.
        """

        # Check displacement vector matches grid dtype
        if displacement_vector.dtype != grid_coords.dtype:
            displacement_vector = displacement_vector.to(grid_coords)

        # Check vector has same dimensions as grid (handle 3D vs 4D)
        dims = grid_coords.shape[1]
        disp_vec = displacement_vector[:dims]

        #  Determine Fault Side (Heaviside Step)
        # If the scalar field is greater then the threshold the result is 1.0 (Hanging wall block)
        # Otherwise it is 0.0 (Footwall block)
        drift_mask = (scalar_field > threshold).float()
        
        # Apply displacement to the points on hanging wall according to displacement vector
        drift = drift_mask.unsqueeze(1) * disp_vec.unsqueeze(0)
        return drift

    def save_internal_state(self):
        """
        Saves all the current parameters to avoid overwriting
        """

        state = {
            'w': self.w.clone(),
            'Transformation_matrix': self.Transformation_matrix.clone(), 
            'Position_G': self.Position_G.clone(),
            'Value_G': self.Value_G.clone(),  
            'Position_G_Modified': self.Position_G_Modified.clone(),
            'ref_layer_points': self.ref_layer_points.clone(),
            'rest_layer_points': self.rest_layer_points.clone(),
            'number_of_layer': self.number_of_layer,
            'number_of_points_per_surface': self.number_of_points_per_surface.clone(),
            'sp_coord': copy.deepcopy(self.sp_coord) # Added sp_coord for safety
        }
        return state

    def load_internal_state(self, state):
        '''
        We can load fault_states and structure_state accordingly

        '''
        self.w = state['w']
        self.Transformation_matrix = state['Transformation_matrix'] 
        self.Position_G = state['Position_G']
        self.Value_G = state['Value_G'] 
        self.Position_G_Modified = state['Position_G_Modified']
        self.ref_layer_points = state['ref_layer_points']
        self.rest_layer_points = state['rest_layer_points']
        self.number_of_layer = state['number_of_layer']
        self.number_of_points_per_surface = state['number_of_points_per_surface']
        self.sp_coord = state['sp_coord']

    def compute_models(self, faults_data_dict, structure_data):

        '''
        Solves Kriging for every fault in the dictionary, then for structure.
        faults_data_dict = {'fault3': input_data, 'fault4': input_data}


        Load fault points -> Solves fault kriginng -> Saves the result
        Load Strcuture points -> Solves Strcuture kriginng -> Saves the result

        
        '''
        
        # print(f"Processing Models")
        
        ######### Solve Each Fault Independently #########
        for fname, f_data in faults_data_dict.items():
            print(f"--- Solving Fault Model: {fname} ---")
            
            self.interface_data(f_data['sp_coord'])
            self.orientation_data(f_data['op_coord'])

            # Transformation matrix for specific fault
      
            if 'transformation_matrix' in f_data:
                self.Transformation_matrix = f_data['transformation_matrix']
            else:
            # Taking default Transformation matrix if not provided
                self.Transformation_matrix = torch.diag(torch.tensor([1,1,1,0],dtype=torch.float32))
                
            # Printing for check
            print(f"{fname} Transformation matrix: {f_data['transformation_matrix']}")

            self.Ge_model() # Solve Kriging
            # Saving current fault model
            self.fault_states[fname] = self.save_internal_state()

        
        ######### Solving Structure Model #########

        # print("Solving Structure Kriging System")
        self.interface_data(structure_data['sp_coord'])
        self.orientation_data(structure_data['op_coord'])
        
        # Transformation matrix for structure
        if 'transformation_matrix' in structure_data:
            self.Transformation_matrix = structure_data['transformation_matrix']
        else:
        # Taking default Transformation matrix if not provided
            self.Transformation_matrix = torch.diag(torch.tensor([1,1,1,0],dtype=torch.float32))

        # Printing for check
        print(f"Strucutre Transformation matrix: {structure_data['transformation_matrix']}")

        self.Ge_model() 
        # Saving structure model
        self.structure_state = self.save_internal_state()
        
        ######### Combine input data for plotting and visualization ########

        # print("Combining Input Data for Visualization")
        # self.sp_coord = {**fault_data['sp_coord'], **structure_data['sp_coord']}
        
        # # Orientation points
        # f_pos = fault_data['op_coord']['Positions']
        # s_pos = structure_data['op_coord']['Positions']
        # self.Position_G = torch.cat([f_pos, s_pos], dim=0)
        
        # # Gradient values
        # f_val = fault_data['op_coord']['Values']
        # s_val = structure_data['op_coord']['Values']
        # self.Value_G = torch.cat([f_val, s_val], dim=0)
        
        # We also need a Dummy 'w' for the Combined state so checking hasattr(self, 'w') passes in the plottng fucntion
        # The actual values don't matter as we reload state in Solution_grid
        self.w = torch.zeros(1) 

        # print("Both models computed successfully")

    def compute_faulted_grid(self, grid_coord=None):
        """
        SEQUENTIAL SOLVER (GemPy Style)
        The grid is updated iteratively. Younger faults deform the grid 
        before the older faults (or structure) are calculated on it.
        """

        if grid_coord is None:
            grid_coord = self.data["Regular"]

        # 1. BACKUP Combined State (Visualization Data)
        # We need to save the current state (which holds all points for plotting)
        # because load_internal_state will overwrite it with single-model data.
        combined_backup = self.save_internal_state()

        # Create a 'working_grid' that accumulates deformation step-by-step
        working_grid = grid_coord.clone()

        total_drift = torch.zeros_like(grid_coord)

        try:

            # ATTENTION!!!: The order of the fault dictionary matters in a sequential model
            # Ensure the loop processes the YOUNGEST fault first.
            # If you don't know the age and have parallel faults don't care about order.

            ###### Step 1: Loop Over faults and Calculate Fault Scalar Field ######
            
            for fname, state in self.fault_states.items():

                # Loading fault data from storage
                self.load_internal_state(state)
                
                # Calculating scalar field on original grid from 4D GemPy super class
                # And capturing the scalar_field output as fault_out

                # IMPORTANT: Update it on the the current working grid
                fault_out, _ = super().Solution_grid(working_grid, section_plot=True, recompute_weights=False)


                ### We calculate fault_scalars to calcualte the drift on each grid point
                ### Or to get volume of points
                if "Regular" in fault_out:
                    fault_scalars = fault_out["Regular"]
                else:
                    # Fallback only if Regular is missing, but avoid 'scalar_ref_points'
                    # Find a key that isn't 'scalar_ref_points'
                    keys = [k for k in fault_out.keys() if k != "scalar_ref_points"]
                    if keys:
                        fault_scalars = fault_out[keys[0]]
                    else:
                        raise ValueError("No grid data returned from Solution_grid")
                
        
                ##### Fault threshold - Average value at fault interface #####
        
                if "scalar_ref_points" in fault_out:
                        # capturing mean of reference points of faults
                        threshold = torch.mean(fault_out["scalar_ref_points"])
                else:
                        threshold = torch.mean(fault_scalars)

                ###### Step 2: Caclualte drift for this fault ######
                
                # Get displacement vector, take 0 if nothing is provided?
                disp = self.fault_displacement_vectors.get(fname, torch.tensor([0.,0.,0.,0.]))
                
                # Here drift is calculated based on where the points are now! (working_grid)
                drift = self.transform_function(fault_scalars, working_grid, threshold, disp)

                ##### Step 3:  Apply Coordinate/Grid Transformation #####
                # Sequential update of working grid wiht drifts
                # We permanently deform the grid for the NEXT step in the loop
                working_grid = working_grid + drift

            
            ###### Step 4: Calculate Structural Scalar Field on Deformed Grid ######
            # Loading structure data from storage
            
            self.load_internal_state(self.structure_state)

            # calcualting solution for structure data on deformed grid
            struct_out, struct_res = super().Solution_grid(working_grid, section_plot=True, recompute_weights=False)
            
            return struct_out, struct_res

        finally:
            # For plotter function:  RESTORE Combined State
            # This ensures that when this function returns, 'self' contains the full
            # combined input data (Position_G, Value_G, etc.) that plot_interactive_section expects.
            self.load_internal_state(combined_backup)

    # 2. The REQUIRED override (The Wrapper)
    def Solution_grid(self, grid_coord=None, section_plot=False, recompute_weights=True):
        """
        Overrides the parent method. Plotter method call this function.
        We simply redirect it to our custom logic.
        """
        return self.compute_faulted_grid(grid_coord)
    
    ####### PLOTTING SCALAR FIELD FOR - FAULT, STRUCTURE, RESULTANT ##############
    

if __name__ == "__main__":

    # Define Gempy model paremters like extent resolution
    extent = [-0.1,1.2,0.1,1.2,-0.1,1.2, 0,5]
    resolution = [50, 50, 50, 2]

    # Initialize Custom Model
    model = GempyFaultModel("Fault_4D_Test", extent, resolution)
    
    # 1. FAULT 1 Data
    fault1_interface_data = {
    'fault1': torch.tensor([
        [500., 500., 500.,   0.],
        [450., 500., 600.,   0.],
        [500., 200., 500.,   0.],
        [450., 200., 600.,   0.],
        [500., 800., 500.,   0.],
        [450., 800., 600.,   0.]
    ])/1000
}
    fault1_orientation_data = {
        'Positions': torch.tensor([
        [500., 500., 500.,   0.]
    ]) / 1000,

        "Values": torch.tensor([
           [0.866, 0.0, 0.5, 0.]
        ])
    }

    fault1_transformation_matrix = torch.diag(torch.tensor([1,1,1,0.01],dtype=torch.float32))
    
    # Fault 1: Moves Hanging Wall UP by 0.2 (Z-axis)
    model.add_fault_displacement('fault1', torch.tensor([0.0, 0.0, 0.2, 0.0]))

    fault1_input = {'sp_coord': fault1_interface_data, 'op_coord': fault1_orientation_data, 
                    'transformation_matrix': fault1_transformation_matrix}

    # 1. FAULT 1 Data
    fault2_interface_data = {
    'fault2': torch.tensor([
        [700., 500., 500.,   0.],
        [650., 500., 600.,   0.],
        [700., 200., 500.,   0.],
        [650., 200., 600.,   0.],
        [700., 800., 500.,   0.],
        [650., 800., 600.,   0.]
    ])/1000
}
    fault2_orientation_data = {
        'Positions': torch.tensor([
        [500., 500., 500.,   0.]
    ]) / 1000,

        "Values": torch.tensor([
           [0.866, 0.0, 0.5, 0.]
        ])
    }

    fault2_transformation_matrix = torch.diag(torch.tensor([1,1,1,0.01],dtype=torch.float32))

    # Fault 1: Moves Hanging Wall UP by 0.2 (Z-axis)
    model.add_fault_displacement('fault2', torch.tensor([0.0, 0.0, 0.3, 0.0]))

    fault2_input = {'sp_coord': fault2_interface_data, 'op_coord': fault2_orientation_data, 
                    'transformation_matrix': fault2_transformation_matrix}

    faults_dict = {
    'fault1': fault1_input,  # Your original fault
    'fault2': fault2_input  # The new parallel fault
    }
    # 2. STRUCTURE Data
    struct_interface_data =  {
        'rock1': torch.tensor([
        [   0.,  200.,  600.,    0.],
        [   0.,  500.,  600.,    0.],
        [   0.,  800.,  600.,    0.],
        [ 200.,  200.,  600.,    0.],
        [ 200.,  500.,  600.,    0.],
        [ 200.,  800.,  600.,    0.],
        [ 800.,  200.,  200.,    0.],
        [ 800.,  500.,  200.,    0.],
        [ 800.,  800.,  200.,    0.],
        [1000.,  200.,  200.,    0.],
        [1000.,  500.,  200.,    0.],
        [1000.,  800.,  200.,    0.]
    ])/1000,
    
    'rock2': torch.tensor([
        [   0.,  200.,  800.,    0.],
        [   0.,  800.,  800.,    0.],
        [ 200.,  200.,  800.,    0.],
        [ 200.,  800.,  800.,    0.],
        [ 800.,  200.,  400.,    0.],
        [ 800.,  800.,  400.,    0.],
        [1000.,  200.,  400.,    0.],
        [1000.,  800.,  400.,    0.]
        ]) / 1000
    }
    struct_orientation_data = {
        'Positions': torch.tensor([
        [100., 500., 800.,   0.],  # rock2
        [100., 500., 600.,   0.],  # rock1
        [900., 500., 400.,   0.],  # rock2
        [900., 500., 200.,   0.],  # rock1

        ]) / 1000,

        "Values": torch.tensor([
            [0., 0., 1., 0.1],
            [0., 0., 1., 0.1],
            [0., 0., 1., -0.1],
            [0., 0., 1., -0.1]
        ])
    }

    struct_transformation_matrix = torch.diag(torch.tensor([1,1,1,0.05],dtype=torch.float32))

    struct_input = {'sp_coord': struct_interface_data, 'op_coord': struct_orientation_data, 'transformation_matrix': struct_transformation_matrix}

    # Compute Models
    model.compute_models(faults_data_dict=faults_dict, structure_data=struct_input)
    
    print("Model computed.")

    #########################################################################
    ###### Uncomment the below code lines for matplotlib visualization ######
    #########################################################################

    ##### FOR 2D matplotlib #####
    # import time
    # for t in [-0.5, 0, 0.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3, 3.5, 4,4.5]:
    #     model.plot_data_section(section={2:0.5, 4:t}, plot_scalar_field = True, plot_input_data=True)
    #     time.sleep(1)


    ##### FOR 3D matplotlib #####
    import time
    for t in [-0.5, 0, 0.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3, 3.5, 4,4.5]:
        model.plot_data_section(section={4:t}, plot_scalar_field = True, plot_input_data=True)
        time.sleep(1)


    #############################################################################################
    ########## Uncomment below for  Interactive Visualization Pyvista below  ####################
    #############################################################################################
    
    ###############################################################
    ########### show/unshow input data using "plot_input_data" argument
    ########### show/unshow surface or interfaces using "only_surface_mode" argument
    ###############################################################

    # --- PLOTTING ---
    # model.plot_interactive_section(plot_input_data=True, only_surface_mode= False)
   