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
        # Dictionaries to store N fault models and N block models
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
        """Evaluates a specific fault's scalar field at given coordinates."""
        self.load_internal_state(self.fault_states[fault_name])
        out, _ = super().Solution_grid(points, section_plot=True, recompute_weights=False)
        return out["Regular"]

    def compute_models(self, faults_data, structure_data):
        '''
        faults_data: Dictionary of multiple faults e.g., {'Fault1': f1_data, 'Fault2': f2_data}
        '''

        # Default transformation matrix if not provided
        default_matrix = torch.diag(torch.tensor([1,1,1,0]))
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
            
            self.Ge_model() 
            # Saving current fault model

            self.fault_states[f_name] = self.save_internal_state()

            # Capture current fault threshold
            out, _ = super().Solution_grid(self.ref_layer_points, section_plot=True, recompute_weights=False)
            # capturing mean of reference points of current fault
            self.fault_thresholds[f_name] = torch.mean(out["scalar_ref_points"])

        ######### Partitioning Data by Boolean Signatures #########
        print("\n- Partitioning Structural Data into Blocks -")
        blocks_struct_data = {}

        def get_signatures(points):
            """Returns a list of boolean tuples representing the 
            block signature for each point."""
            if len(points) == 0: return []

            sigs = [] #signatures
            
            for f_name in self.fault_names:
                scalars = self.eval_fault_at_points(f_name, points).squeeze()
                if scalars.dim() == 0: scalars = scalars.unsqueeze(0)
                mask = scalars > self.fault_thresholds[f_name]
                sigs.append(mask)

            
            
            # Stack and convert to rows of tuples
            sigs_stacked = torch.stack(sigs, dim=1)
            return [tuple(row.tolist()) for row in sigs_stacked]

        # Route Interface Points to correct block dictionary
        for layer_name, points in structure_data['sp_coord'].items():
            sigs = get_signatures(points)
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
        
        for i, sig in enumerate(sigs):
            if sig not in blocks_struct_data:
                blocks_struct_data[sig] = {'sp_coord': {}, 'op_coord': {'Positions': [], 'Values': []}}
            blocks_struct_data[sig]['op_coord']['Positions'].append(op_pos[i])
            blocks_struct_data[sig]['op_coord']['Values'].append(op_val[i])

        # Convert grouped lists back to PyTorch tensors
        for sig in blocks_struct_data:
            for layer_name in blocks_struct_data[sig]['sp_coord']:
                blocks_struct_data[sig]['sp_coord'][layer_name] = torch.stack(blocks_struct_data[sig]['sp_coord'][layer_name])
            if len(blocks_struct_data[sig]['op_coord']['Positions']) > 0:
                blocks_struct_data[sig]['op_coord']['Positions'] = torch.stack(blocks_struct_data[sig]['op_coord']['Positions'])
                blocks_struct_data[sig]['op_coord']['Values'] = torch.stack(blocks_struct_data[sig]['op_coord']['Values'])

        ######### Solve Kriging for Each Populated Block #########
        self.block_states = {}
        for sig, data in blocks_struct_data.items():
            has_interfaces = len(data['sp_coord']) > 0
            has_orientations = len(data['op_coord']['Positions']) > 0
            
            if has_interfaces and has_orientations:
                print(f"\n--- Solving Structural Block {sig} ---")
                self.interface_data(data['sp_coord'])
                self.orientation_data(data['op_coord'])
                self.Transformation_matrix = structure_data.get('transformation_matrix', default_matrix)
                self.Ge_model()
                self.block_states[sig] = self.save_internal_state()
            else:
                print(f"\nWarning: Block {sig} skipped (Needs at least 1 interface and 1 orientation point in this block).")

        # print(blocks_struct_data)
        # print(self.block_states)


        ######### Combine Original Data for Visualization #########
        self.sp_coord = structure_data['sp_coord'].copy()
        for f_data in faults_data.values():
            self.sp_coord.update(f_data['sp_coord'])
            
        all_op_pos = [structure_data['op_coord']['Positions']] + [f['op_coord']['Positions'] for f in faults_data.values()]
        all_op_val = [structure_data['op_coord']['Values']] + [f['op_coord']['Values'] for f in faults_data.values()]
        
        self.Position_G = torch.cat(all_op_pos, dim=0)
        self.Value_G = torch.cat(all_op_val, dim=0)

        # For plotter
        self.w = torch.zeros(1) 

    def compute_faulted_grid(self, grid_coord=None):

        """
        Making the grid for plotting
        """

        if grid_coord is None:
            grid_coord = self.data["Regular"]

        combined_backup = self.save_internal_state()

        try:
            # Evaluate all fault boundaries on the 3D Grid and
            # Creating all fault masks
            grid_fault_masks = {}
            for f_name in self.fault_names:
                self.load_internal_state(self.fault_states[f_name])
                out, _ = super().Solution_grid(grid_coord, section_plot=True, recompute_weights=False)
                grid_fault_masks[f_name] = out["Regular"] > self.fault_thresholds[f_name]

            # Evaluate all populated structural blocks independently assuming fault do not exit
            block_outputs, block_res = {}, {}
            for sig in self.block_states:
                self.load_internal_state(self.block_states[sig])
                out, res = super().Solution_grid(grid_coord, section_plot=True, recompute_weights=False)
                block_outputs[sig] = out
                block_res[sig] = res

            if not block_outputs:
                raise ValueError("No valid structural blocks were computed!")

            # 3. Stitch blocks together based on Boolean Signatures
            final_out, final_res = {}, {}
            template_sig = list(block_outputs.keys())[0]
            
            # Initialize empty tensors
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
                    if expected:
                        block_mask = block_mask & grid_fault_masks[f_name]
                    else:
                        block_mask = block_mask & (~grid_fault_masks[f_name])

                # Fill data into final output where the mask is True
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
        return self.compute_faulted_grid(grid_coord)


if __name__ == "__main__":
    extent = [0, 3, 0, 1, 0, 1, 0, 5]
    resolution = [50, 50, 50, 2]

    # Initialize Custom Model
    model = GempyMultiFaultModel("Fault_4D_Test", extent, resolution)

    # =========================================
    #             1. FAULTS DATA
    # =========================================
    
    # --- Original Fault (from the CSV) ---
    fault_interface = {
        'fault': torch.tensor([
            [2000.0, 0.0, 400.0, 0.0],
            [2000.0, 500.0, 400.0, 0.0],
            [2000.0, 1000.0, 400.0, 0.0]
        ]) / 1000
    }
    fault_orientation = {
        'Positions': torch.tensor([[2000.0, 500.0, 410.0, 0.0],
                                   
                                   [1990.0, 500.0, 420.0, 0.0],

                                   [2010.0, 500.0, 390.0, 0.0],
                                   
                                   ]) / 1000,
        "Values": torch.tensor([[0.819, 0.000, 0.574, 0.0],
                                
                                [0.819, 0.000, 0.574, -0.05],

                                [0.819, 0.000, 0.574, 0.05],
                                
                                ])
    }
    fault_trans = torch.diag(torch.tensor([1, 1, 1, 0.05], dtype=torch.float32))

    # --- NEW Fault 0 ---
    fault0_interface = {
        'fault0': torch.tensor([
            [2750.0, 0.0, 400.0, 0.0],
            [2750.0, 500.0, 400.0, 0.0],
            [2750.0, 1000.0, 400.0, 0.0]
        ]) / 1000
    }
    fault0_orientation = {
        # Z shifted to 410.0 to prevent singular matrix!
        'Positions': torch.tensor([[2750.0, 500.0, 410.0, 0.0],
                                   
                                   [2740.0, 500.0, 420.0, 0.0],
                                   
                                   [2760.0, 500.0, 400.0, 0.0]]) / 1000,

        "Values": torch.tensor([[0.800, 0.000, 0.600, 0.0],
                                
                                [0.800, 0.000, 0.600, -0.05],

                                [0.800, 0.000, 0.600, 0.05]
                                ])
    }
    fault0_trans = torch.diag(torch.tensor([1, 1, 1, 0.05], dtype=torch.float32))

    # Combine both faults into the dictionary
    faults_dict = {
        'FAULT': {'sp_coord': fault_interface, 'op_coord': fault_orientation, 'transformation_matrix': fault_trans},
        'FAULT0': {'sp_coord': fault0_interface, 'op_coord': fault0_orientation, 'transformation_matrix': fault0_trans}
    }

    # =========================================
    #           2. STRUCTURE DATA
    # =========================================
    struct_interface_data = {
        'rock1': torch.tensor([
            [0.0, 0.0, 250.0, 0.0], [0.0, 500.0, 250.0, 0.0], [0.0, 1000.0, 250.0, 0.0],
            [150.0, 0.0, 400.0, 0.0], [150.0, 500.0, 400.0, 0.0], [150.0, 1000.0, 400.0, 0.0],
            [300.0, 0.0, 550.0, 0.0], [300.0, 500.0, 550.0, 0.0], [300.0, 1000.0, 550.0, 0.0],
            [450.0, 0.0, 400.0, 0.0], [450.0, 500.0, 400.0, 0.0], [450.0, 1000.0, 400.0, 0.0],
            [600.0, 0.0, 250.0, 0.0], [600.0, 500.0, 250.0, 0.0], [600.0, 1000.0, 250.0, 0.0],
            [750.0, 0.0, 350.0, 0.0], [750.0, 500.0, 350.0, 0.0], [750.0, 1000.0, 350.0, 0.0],
            [900.0, 0.0, 550.0, 0.0], [900.0, 500.0, 550.0, 0.0], [900.0, 1000.0, 550.0, 0.0],
            [1450.0, 0.0, 550.0, 0.0], [1450.0, 500.0, 550.0, 0.0], [1450.0, 1000.0, 550.0, 0.0],
            [1600.0, 0.0, 350.0, 0.0], [1600.0, 500.0, 350.0, 0.0], [1600.0, 1000.0, 350.0, 0.0],
            [1750.0, 0.0, 250.0, 0.0], [1750.0, 500.0, 250.0, 0.0], [1750.0, 1000.0, 250.0, 0.0],
            [1850.0, 0.0, 350.0, 0.0], [1850.0, 500.0, 350.0, 0.0], [1850.0, 1000.0, 350.0, 0.0],
            [2200.0, 0.0, 350.0, 0.0], [2200.0, 500.0, 350.0, 0.0], [2200.0, 1000.0, 350.0, 0.0],
            [2300.0, 0.0, 450.0, 0.0], [2300.0, 500.0, 450.0, 0.0], [2300.0, 1000.0, 450.0, 0.0],
            [1050.0, 0.0, 799.0, 0.0], [1050.0, 500.0, 799.0, 0.0], [1050.0, 1000.0, 799.0, 0.0],
            [1300.0, 0.0, 799.0, 0.0], [1300.0, 500.0, 799.0, 0.0], [1300.0, 1000.0, 799.0, 0.0],

            [2950.0, 0.0, 350.0, 0.0], [2950.0, 500.0, 350.0, 0.0], [2950.0, 1000.0, 350.0, 0.0]
        ]) / 1000,
        
        'rock2': torch.tensor([
            [0.0, 0.0, 450.0, 0.0], [0.0, 500.0, 450.0, 0.0], [0.0, 1000.0, 450.0, 0.0],
            [150.0, 0.0, 600.0, 0.0], [150.0, 500.0, 600.0, 0.0], [150.0, 1000.0, 600.0, 0.0],
            [300.0, 0.0, 750.0, 0.0], [300.0, 500.0, 750.0, 0.0], [300.0, 1000.0, 750.0, 0.0],
            [450.0, 0.0, 600.0, 0.0], [450.0, 500.0, 600.0, 0.0], [450.0, 1000.0, 600.0, 0.0],
            [600.0, 0.0, 450.0, 0.0], [600.0, 500.0, 450.0, 0.0], [600.0, 1000.0, 450.0, 0.0],
            [750.0, 0.0, 600.0, 0.0], [750.0, 500.0, 600.0, 0.0], [750.0, 1000.0, 600.0, 0.0],
            [850.0, 0.0, 800.0, 0.0], [850.0, 500.0, 800.0, 0.0], [850.0, 1000.0, 800.0, 0.0],
            [1500.0, 0.0, 800.0, 0.0], [1500.0, 500.0, 800.0, 0.0], [1500.0, 1000.0, 800.0, 0.0],
            [1600.0, 0.0, 600.0, 0.0], [1600.0, 500.0, 600.0, 0.0], [1600.0, 1000.0, 600.0, 0.0],
            [1750.0, 0.0, 450.0, 0.0], [1750.0, 500.0, 450.0, 0.0], [1750.0, 1000.0, 450.0, 0.0],
            [1850.0, 0.0, 550.0, 0.0], [1850.0, 500.0, 550.0, 0.0], [1850.0, 1000.0, 550.0, 0.0],
            [2200.0, 0.0, 550.0, 0.0], [2200.0, 500.0, 550.0, 0.0], [2200.0, 1000.0, 550.0, 0.0],
            [2300.0, 0.0, 650.0, 0.0], [2300.0, 500.0, 650.0, 0.0], [2300.0, 1000.0, 650.0, 0.0],

            [2950.0, 0.0, 500.0, 0.0], [2950.0, 500.0, 500.0, 0.0], [2950.0, 1000.0, 500.0, 0.0]
        ]) / 1000,
    }

    struct_orientation_data = {
        'Positions': torch.tensor([
            [0.0, 500.0, 460.0, 0.0],
            [0.0, 500.0, 260.0, 0.0],
            [0.0, 0.0, 460.0, 0.0],
            [0.0, 0.0, 260.0, 0.0],
            [0.0, 1000.0, 460.0, 0.0],
            [0.0, 1000.0, 260.0, 0.0],

            # --- ADDED: Middle Block Orientation (2000 < X < 2750) ---
            [2200.0, 500.0, 460.0, 0.0],
            
            # --- ADDED: 3rd Block Orientation (X > 2750) ---
            [2950.0, 500.0, 460.0, 0.0]
        ]) / 1000,
        "Values": torch.tensor([
            [-0.423, -0.000, 0.906, 0.0],
            [-0.423, -0.000, 0.906, 0.0],
            [-0.423, -0.000, 0.906, 0.0],
            [-0.423, -0.000, 0.906, 0.0],
            [-0.423, -0.000, 0.906, 0.0],
            [-0.423, -0.000, 0.906, 0.0],

            [-0.423, -0.000, 0.906, -0.07],

            [-0.423, -0.000, 0.906, -0.20]


        ])
    }
    
    struct_transformation_matrix = torch.diag(torch.tensor([1, 1, 1, 0.05], dtype=torch.float32))

    struct_input = {
        'sp_coord': struct_interface_data, 
        'op_coord': struct_orientation_data,
        'transformation_matrix': struct_transformation_matrix
    }

    # Run computation
    model.compute_models(faults_data=faults_dict, structure_data=struct_input)
    print("\nModel computed successfully!")
    
    
    # --- PLOTTING ---
    #########################################################################
    ###### Uncomment the below code lines for matplotlib visualization ######
    #########################################################################

    #### FOR 2D matplotlib #####
    # import time
    # for t in [-0.5, 0, 0.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3,3.5, 4, 4.5]:
    #     model.plot_data_section(section={2:0.5, 4:t}, plot_scalar_field = True, plot_input_data=True)
    #     time.sleep(1)


    #### FOR 3D matplotlib #####
    # import time
    # for t in [-0.5, 0, 0.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3, 3.5, 4,4.5]:
    #     model.plot_data_section(section={4:t}, plot_scalar_field = False, plot_input_data=False)
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
    model.plot_interactive_section(plot_input_data=True, only_surface_mode= False)