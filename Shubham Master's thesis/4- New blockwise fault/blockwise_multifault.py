import torch
import numpy as np
import copy
import matplotlib.pyplot as plt

# Import Gempy class from pyvista_new.py or withbasisfunction.py(with universal term)
from pyvista_new import Gempy

class GempyMultiFaultModel(Gempy):
    """
    Handles multiple faults dynamically using Boolean Block Signatures.
    
    Extension of 4D Gempy class to handle faults.
    Block-wise Solving.
    Splitting the structural dataset based on the fault surface 
    and interpolate both sides independently.

    """

    def __init__(self, project_name, extent, resolution):

        ## calling initialization of the parent 'Gempy' class
        super().__init__(project_name, extent, resolution)
        
        # Storing independent kriging systems
        # Dictionaries to store N fault models + thresholds and (N+1) block models
        self.fault_states = {}
        self.fault_thresholds = {}

        self.block_states = {}  # Keys will be boolean tuples e.g., (True, False)

    def save_internal_state(self):
        """Saves all parameters to avoid overwriting between
        different systems.
        """
        return {
            'w': self.w.clone(),
            'Transformation_matrix': self.Transformation_matrix.clone(), 
            'Position_G': self.Position_G.clone(),
            'Value_G': self.Value_G.clone(),  
            'Position_G_Modified': self.Position_G_Modified.clone(),
            'ref_layer_points': self.ref_layer_points.clone(),
            'rest_layer_points': self.rest_layer_points.clone(),
            'number_of_layer': self.number_of_layer,
            'number_of_points_per_surface': self.number_of_points_per_surface.clone(),
            'sp_coord': copy.deepcopy(self.sp_coord)
        }

    def load_internal_state(self, state):
        """
         We can load specific kriging system paratmeters.
        """
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

    def eval_fault_at_points(self, fault_name, points):
        """ 
        Kind of a helper function.
        
        Evaluates a specific fault's scalar field at given coordinates
        """
        self.load_internal_state(self.fault_states[fault_name])
        out, _ = super().Solution_grid(points, section_plot=True, recompute_weights=False)
        return out["Regular"]

    def compute_models(self, faults_data, structure_data):
        '''
        faults_data: Dictionary of multiple faults e.g., {'Fault1': f1_data, 'Fault2': f2_data}
        structure_data: As in previous file
        '''

        # Default transformation matrix if not provided
        default_matrix = torch.diag(torch.tensor([1,1,1,0]))
        #Extracting all fault names provided
        self.fault_names = list(faults_data.keys())
        
        ######### Solving All Fault Models #########

        # Runs through each fault we provided
        for f_name in self.fault_names:
            print(f"\n- Solving Fault: {f_name} -")

            # Choose the present fault to be evaluated
            f_data = faults_data[f_name]


            self.interface_data(f_data['sp_coord'])
            self.orientation_data(f_data['op_coord'])
            self.Transformation_matrix = f_data.get('transformation_matrix', default_matrix)
            
            # Solving currect fault model
            self.Ge_model() 

            # Saving current fault model
            self.fault_states[f_name] = self.save_internal_state()

            # Capture current fault threshold
            out, _ = super().Solution_grid(self.ref_layer_points, section_plot=True, recompute_weights=False)
            # capturing mean of reference points of current fault
            self.fault_thresholds[f_name] = torch.mean(out["scalar_ref_points"])

        ######### Partitioning Data by using Boolean Signatures #########
        print("\n- Partitioning Structural Data into Blocks -")

        # Dictionary to store fault separated blocks
        blocks_struct_data = {}

        def get_signatures(points):
            """
            Helper function

            "Takes a list of structural data points and evaluates them against every fault solution"

            Returns a list of boolean tuples representing the 
            block signature for each point
            
            """
            if len(points) == 0: return []

            sigs = [] #signatures
            
            for f_name in self.fault_names:
                scalars = self.eval_fault_at_points(f_name, points).squeeze()
                if scalars.dim() == 0: scalars = scalars.unsqueeze(0)
                mask = scalars > self.fault_thresholds[f_name]
                
                # Appending True for false for all points according to current fault (f_name)
                sigs.append(mask)

            
            
            # Stack boolean signatures and convert to rows of tuples
            sigs_stacked = torch.stack(sigs, dim=1)
            return [tuple(row.tolist()) for row in sigs_stacked]

        # Marking Interface Points to correct block dictionary(booleans)
        for layer_name, points in structure_data['sp_coord'].items():
            sigs = get_signatures(points)
            
            #Check
            # print(f"Signatues are: {sigs}")
         
            for i, sig in enumerate(sigs):

                # Drops a value in blocks_struct_data which tells where the
                # structural point actually is (above, below or between faults)
                if sig not in blocks_struct_data:
                    blocks_struct_data[sig] = {'sp_coord': {}, 'op_coord': {'Positions': [], 'Values': []}}
                if layer_name not in blocks_struct_data[sig]['sp_coord']:
                    blocks_struct_data[sig]['sp_coord'][layer_name] = []
                blocks_struct_data[sig]['sp_coord'][layer_name].append(points[i])

        ## SAME FOR ORIENTATION POINTS
        # Route Orientation Points to correct block dictionary
        op_pos = structure_data['op_coord']['Positions']
        op_val = structure_data['op_coord']['Values']
        sigs = get_signatures(op_pos)
        
        #Check
        # print(f"Orientation point sigantures: {sigs}")


        for i, sig in enumerate(sigs):
            if sig not in blocks_struct_data:
                blocks_struct_data[sig] = {'sp_coord': {}, 'op_coord': {'Positions': [], 'Values': []}}
            blocks_struct_data[sig]['op_coord']['Positions'].append(op_pos[i])
            blocks_struct_data[sig]['op_coord']['Values'].append(op_val[i])

        # Check
        # print(f"Block data dictionary: {blocks_struct_data}")

        # Convert grouped lists back to PyTorch tensors
        for sig in blocks_struct_data:
            for layer_name in blocks_struct_data[sig]['sp_coord']:
                blocks_struct_data[sig]['sp_coord'][layer_name] = torch.stack(blocks_struct_data[sig]['sp_coord'][layer_name])
            if len(blocks_struct_data[sig]['op_coord']['Positions']) > 0:
                blocks_struct_data[sig]['op_coord']['Positions'] = torch.stack(blocks_struct_data[sig]['op_coord']['Positions'])
                blocks_struct_data[sig]['op_coord']['Values'] = torch.stack(blocks_struct_data[sig]['op_coord']['Values'])

        # print(f"Block data tensor: {blocks_struct_data}")

        ######### Solve Kriging for Each Populated Block #########

        #For Storing block states/parameters
        self.block_states = {}

        for sig, data in blocks_struct_data.items():
            has_interfaces = len(data['sp_coord']) > 0
            has_orientations = len(data['op_coord']['Positions']) > 0
            
            if has_interfaces and has_orientations:
                print(f"\n--- Solving Structural Block {sig} ---")
                self.interface_data(data['sp_coord'])
                self.orientation_data(data['op_coord'])
                self.Transformation_matrix = structure_data.get('transformation_matrix', default_matrix)

                #Solving each block
                self.Ge_model()
                #Saving each block with signature
                self.block_states[sig] = self.save_internal_state()
            else:
                print(f"\nWarning: Block {sig} skipped (Needs at least 1 interface and 1 orientation point in this block).")

        # Saved data
        # print(self.block_states)


        ######### Combine Original Data for Visualization #########
        self.sp_coord = structure_data['sp_coord'].copy()
        for f_data in faults_data.values():
            self.sp_coord.update(f_data['sp_coord'])
            
        all_op_pos = [structure_data['op_coord']['Positions']] + [f['op_coord']['Positions'] for f in faults_data.values()]
        all_op_val = [structure_data['op_coord']['Values']] + [f['op_coord']['Values'] for f in faults_data.values()]
        
        self.Position_G = torch.cat(all_op_pos, dim=0)
        self.Value_G = torch.cat(all_op_val, dim=0)

        # Default weights for first plotter
        self.w = torch.zeros(1) 

    def compute_faulted_grid(self, grid_coord=None):

        """
        Making the grid for plotting
        """

        if grid_coord is None:
            grid_coord = self.data["Regular"]

        combined_backup = self.save_internal_state()

        try:
            # Evaluate all fault boundaries/model on the 3D Grid and
            
            grid_fault_masks = {}
            for f_name in self.fault_names:
                self.load_internal_state(self.fault_states[f_name])

                # Calculating scalar field on whole grid using current fault model weights
                out, _ = super().Solution_grid(grid_coord, section_plot=True, recompute_weights=False)
                
                # Creating fault masks for each fault on the whole grid
                grid_fault_masks[f_name] = out["Regular"] > self.fault_thresholds[f_name]

            #Check
            # print(f"Grid_fault_masks are: {grid_fault_masks}")
            # print(f"Grid_fault_masks are: {grid_fault_masks["fault1"].shape}")
           
            # Evaluate all populated structural blocks independently assuming fault do not exit
            block_outputs, block_res = {}, {}
            for sig in self.block_states:
                self.load_internal_state(self.block_states[sig])
                out, res = super().Solution_grid(grid_coord, section_plot=True, recompute_weights=False)
                block_outputs[sig] = out
                block_res[sig] = res

            # print(f"Block outputs:{block_outputs}")


            if not block_outputs:
                raise ValueError("No valid structural blocks were computed!")

            # Stitch blocks together based on Boolean Signatures for plotting
            # Initializing final output
            final_out, final_res = {}, {}

            #Leftmost block signature
            template_sig = list(block_outputs.keys())[0]
            # print(template_sig)

            # Initialize empty tensors for the grid
            for k in block_outputs[template_sig].keys():
                final_out[k] = block_outputs[template_sig][k] if k == 'scalar_ref_points' else torch.zeros_like(block_outputs[template_sig][k])
            for k in block_res[template_sig].keys():
                final_res[k] = block_res[template_sig][k] if k == 'ref_points' else torch.zeros_like(block_res[template_sig][k])


            # Apply masking
            for sig, out_dict in block_outputs.items():
                # Build compound mask for this specific block
                block_mask = torch.ones_like(grid_fault_masks[self.fault_names[0]], dtype=torch.bool)
                for i, f_name in enumerate(self.fault_names):
                    expected = sig[i]

                    # Using both block boolean signature and grid_fault_mask 
                    # And taking common regions
                    if expected:
                        block_mask = block_mask & grid_fault_masks[f_name]
                    else:
                        block_mask = block_mask & (~grid_fault_masks[f_name])

                # Fill data into final output where the common mask is True
                for k in out_dict.keys():
                    if k != 'scalar_ref_points':
                        final_out[k] = torch.where(block_mask, out_dict[k], final_out[k])
                for k in block_res[sig].keys():
                    if k != 'ref_points':
                        final_res[k] = torch.where(block_mask, block_res[sig][k], final_res[k])

            return final_out, final_res

        finally:
            self.load_internal_state(combined_backup)

    def Solution_grid(self, grid_coord=None, section_plot=False, recompute_weights=True):

        """ 
        Overriding function.
        
        Function to stop new Solution_grid calculation inside the plotting method
        """
        return self.compute_faulted_grid(grid_coord)


if __name__ == "__main__":
    # Define Gempy model paremters like extent resolution
    extent = [-0.1, 1.2, 0.1, 1.2, -0.1, 1.2, 0, 5]
    resolution = [50, 50, 50, 2]

    # Initialize Custom Model
    model = GempyMultiFaultModel("Fault_4D_Test", extent, resolution)
    
    # --- 1. FAULT DATA ---
    fault1_interface_data = {
        'fault1': torch.tensor([
            [500., 500., 500.,   0.],
            [450., 500., 600.,   0.],
            [500., 200., 500.,   0.],
            [450., 200., 600.,   0.],
            [500., 800., 500.,   0.],
            [450., 800., 600.,   0.]
        ]) / 1000
    }
    fault1_orientation_data = {
        'Positions': torch.tensor([[500., 500., 500.,   0.]]) / 1000,
        "Values": torch.tensor([[0.866, 0.0, 0.5, 0.]])
    }
    fault1_transformation_matrix = torch.diag(torch.tensor([1, 1, 1, 0.01], dtype=torch.float32))

    fault2_interface_data = {
        'fault2': torch.tensor([
            [700., 500., 500.,   0.],
            [650., 500., 600.,   0.],
            [700., 200., 500.,   0.],
            [650., 200., 600.,   0.],
            [700., 800., 500.,   0.],
            [650., 800., 600.,   0.]
        ]) / 1000
    }
    fault2_orientation_data = {
        'Positions': torch.tensor([[700., 500., 500.,   0.]]) / 1000,
        "Values": torch.tensor([[0.866, 0.0, 0.5, 0.]])
    }
    fault2_transformation_matrix = torch.diag(torch.tensor([1, 1, 1, 0.01], dtype=torch.float32))

    # Group into the required 'faults_dict' structure
    faults_dict = {
        'fault1': {
            'sp_coord': fault1_interface_data, 
            'op_coord': fault1_orientation_data, 
            'transformation_matrix': fault1_transformation_matrix
        },
        'fault2': {
            'sp_coord': fault2_interface_data, 
            'op_coord': fault2_orientation_data, 
            'transformation_matrix': fault2_transformation_matrix
        }
    }

    # --- 2. STRUCTURE DATA ---
    struct_interface_data =  {
        'rock1': torch.tensor([
            # Left Block (X < 450)
            [  0.,  200.,  600.,    0.], [  0.,  500.,  600.,    0.], [  0.,  800.,  600.,    0.],
            [ 200.,  200.,  600.,    0.], [ 200.,  500.,  600.,    0.], [ 200.,  800.,  600.,    0.],
            
            # Center Block (X = ~550). I shifted Z to 400 to reflect an offset
            [ 550.,  200.,  400.,    0.], [ 550.,  500.,  400.,    0.], [ 550.,  800.,  400.,    0.],
            
            # Right Block (X > 700)
            [ 800.,  200.,  200.,    0.], [ 800.,  500.,  200.,    0.], [ 800.,  800.,  200.,    0.],
            [1000.,  200.,  200.,    0.], [1000.,  500.,  200.,    0.], [1000.,  800.,  200.,    0.]
        ]) / 1000,
        
        'rock2': torch.tensor([
            # Left Block
            [  0.,  200.,  800.,    0.], [  0.,  800.,  800.,    0.],
            [ 200.,  200.,  800.,    0.], [ 200.,  800.,  800.,    0.],
            
            # Center Block (Shifted Z to 600 to reflect offset)
            [ 550.,  200.,  600.,    0.], [ 550.,  800.,  600.,    0.],
            
            # Right Block
            [ 800.,  200.,  400.,    0.], [ 800.,  800.,  400.,    0.],
            [1000.,  200.,  400.,    0.], [1000.,  800.,  400.,    0.]
        ]) / 1000
    }

    struct_orientation_data = {
        'Positions': torch.tensor([
            # Left Block
            [100., 500., 800.,   0.],  # rock2
            [100., 500., 600.,   0.],  # rock1
            
            # Center Block (ADDED to prevent singular matrix!)
            [550., 500., 600.,   0.],  # rock2
            [550., 500., 400.,   0.],  # rock1
            
            # Right Block
            [900., 500., 400.,   0.],  # rock2
            [900., 500., 200.,   0.],  # rock1
        ]) / 1000,

        "Values": torch.tensor([
            [0., 0., 1., 0.1],  # Left rock2
            [0., 0., 1., 0.1],  # Left rock1
            [0., 0., 1., 0.0],  # Center rock2 (Flat)
            [0., 0., 1., 0.0],  # Center rock1 (Flat)
            [0., 0., 1., -0.1], # Right rock2
            [0., 0., 1., -0.1]  # Right rock1
        ])
    }
    
    struct_transformation_matrix = torch.diag(torch.tensor([1, 1, 1, 0.01], dtype=torch.float32))

    # Group structure data into dict
    struct_input = {
        'sp_coord': struct_interface_data, 
        'op_coord': struct_orientation_data,
        'transformation_matrix': struct_transformation_matrix
    }

    # Run computation and plot
    model.compute_models(faults_data=faults_dict, structure_data=struct_input)
    
    
    # --- PLOTTING ---
    #########################################################################
    ###### Uncomment the below code lines for matplotlib visualization ######
    #########################################################################

    #### FOR 2D matplotlib #####
    import time
    for t in [-0.5, 0, 0.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3]:
        model.plot_data_section(section={2:0.5, 4:t}, plot_scalar_field = True, plot_input_data=True)
        time.sleep(1)


    #### FOR 3D matplotlib #####
    # import time
    # for t in [-0.5, 0, 0.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3, 3.5, 4,4.5]:
    #     model.plot_data_section(section={4:t}, plot_scalar_field = True, plot_input_data=True)
    #     time.sleep(1)


    #############################################################################################
    ########## Uncomment below for  Interactive Visualization Pyvista below  ####################
    #############################################################################################
    
    ###############################################################
    ########### show/unshow input data using "plot_input_data" argument
    ########### show/unshow surface or interfaces using "only_surface_mode" argument
    ###############################################################

    # 

    # --- PLOTTING ---
    # model.plot_interactive_section(plot_input_data=True, only_surface_mode= False)