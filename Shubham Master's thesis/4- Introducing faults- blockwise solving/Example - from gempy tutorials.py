import torch
import numpy as np
import copy
import matplotlib.pyplot as plt

# Import Gempy class from pyvista_gempy_local_anisotropy.py or with_universal_basis_term.py(with universal term)
from pyvista_gempy_local_anisotropy import Gempy

class GempyMultiFaultModel(Gempy):
    """
    Handles multiple faults dynamically using Boolean Block Signatures
    
    It utilizes a "Block-wise Solving" approach:
    instead of solving a single massive, non-linear system for the entire faulted 
    domain, it computes fault surfaces independently, partitions the domain into 
    topologically distinct blocks, and solves independent Universal Co-Kriging (UCK) 
    systems for each block.

    """

    def __init__(self, project_name, extent, resolution):

        # Initialization of the parent 'Gempy' class
        super().__init__(project_name, extent, resolution)
        
        # Dictionaries to store independent kriging system states
        # fault_states: Stores the weights and matrices for N fault models
        # fault_thresholds: Stores the scalar isovalues defining the fault surfaces
        self.fault_states = {}
        self.fault_thresholds = {}
        
        # block_states: Stores the solved kriging systems for each structurally isolated geological block
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
        Restores a previously saved kriging system state. Used during the 
        final grid evaluation
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
        A helper function.
        Evaluates the scalar drift field of a specific fault at given spatial coordinates.
        This determines whether a spatial point lies in the hanging wall or footwall 
        of this specific fault.
    
        """
        self.load_internal_state(self.fault_states[fault_name])
        out, _ = super().Solution_grid(points, section_plot=True, recompute_weights=False)
        return out["Regular"]

    def compute_models(self, faults_data, structure_data):
        '''
        Core engine of the Block-wise Solving approach.
        
        Args:
            faults_data: Dictionary containing interface and orientation data for multiple faults.
            structure_data: Dictionary containing stratigraphy data.
        '''

        # Default transformation matrix if not provided
        default_matrix = torch.diag(torch.tensor([1,1,1,0]))
        # Extracting all fault's from provided input
        self.fault_names = list(faults_data.keys())
        
        # --- Solving Independent Fault Models --- #

        # Each fault is solved independently
        for f_name in self.fault_names:
            print(f"\n- Solving Fault: {f_name} -")

            # Choose the current fault to be evaluated
            f_data = faults_data[f_name]

            self.interface_data(f_data['sp_coord'])
            self.orientation_data(f_data['op_coord'])
            self.Transformation_matrix = f_data.get('transformation_matrix', default_matrix)
            
            # Solving Universal Co-Kriging for the current fault surface
            self.Ge_model() 

            # Save the solved fault state to the dictionary
            self.fault_states[f_name] = self.save_internal_state()

            # Determine the exact scalar threshold that represents the fault plane
            out, _ = super().Solution_grid(self.ref_layer_points, section_plot=True, recompute_weights=False)
            self.fault_thresholds[f_name] = torch.mean(out["scalar_ref_points"])

        # --- Partitioning Data by using Boolean Signatures --- #
        print("\n- Partitioning Structural Data into Blocks -")

        # Dictionary to store fault separated blocks
        blocks_struct_data = {}

        def get_signatures(points):
            """
            A Helper function

            "Takes a list of structural data points and evaluates them against every fault plane"

            Example: For 2 faults, a point might have signature (True, False), 
            meaning it is > threshold for Fault 1, and < threshold for Fault 2.
            
            """
            if len(points) == 0: return []

            sigs = [] #signatures
            
            # Evaluate points against every fault field
            for f_name in self.fault_names:
                scalars = self.eval_fault_at_points(f_name, points).squeeze()
                if scalars.dim() == 0: scalars = scalars.unsqueeze(0)
                mask = scalars > self.fault_thresholds[f_name]
                sigs.append(mask)
            
            # Stack N boolean arrays and convert to a list of N-dimensional tuples
            sigs_stacked = torch.stack(sigs, dim=1)
            return [tuple(row.tolist()) for row in sigs_stacked]

        # Route Stratigraphic Interface Points to their respective blocks
        for layer_name, points in structure_data['sp_coord'].items():
            sigs = get_signatures(points)
        
            for i, sig in enumerate(sigs):
                if sig not in blocks_struct_data:
                    blocks_struct_data[sig] = {'sp_coord': {}, 'op_coord': {'Positions': [], 'Values': []}}
                if layer_name not in blocks_struct_data[sig]['sp_coord']:
                    blocks_struct_data[sig]['sp_coord'][layer_name] = []
                blocks_struct_data[sig]['sp_coord'][layer_name].append(points[i])

    
        # Route Orientation Points (dip/strike) to their respective blocks
        op_pos = structure_data['op_coord']['Positions']
        op_val = structure_data['op_coord']['Values']
        sigs = get_signatures(op_pos)
        
        for i, sig in enumerate(sigs):
            if sig not in blocks_struct_data:
                blocks_struct_data[sig] = {'sp_coord': {}, 'op_coord': {'Positions': [], 'Values': []}}
            blocks_struct_data[sig]['op_coord']['Positions'].append(op_pos[i])
            blocks_struct_data[sig]['op_coord']['Values'].append(op_val[i])


        # Convert partitioned lists back to PyTorch tensors for co-kriging math
        for sig in blocks_struct_data:
            for layer_name in blocks_struct_data[sig]['sp_coord']:
                blocks_struct_data[sig]['sp_coord'][layer_name] = torch.stack(blocks_struct_data[sig]['sp_coord'][layer_name])
            if len(blocks_struct_data[sig]['op_coord']['Positions']) > 0:
                blocks_struct_data[sig]['op_coord']['Positions'] = torch.stack(blocks_struct_data[sig]['op_coord']['Positions'])
                blocks_struct_data[sig]['op_coord']['Values'] = torch.stack(blocks_struct_data[sig]['op_coord']['Values'])

        # --- Solve Kriging for Each Isolated Block --- #

        # Dictionary for storing block states/parameters
        self.block_states = {}

        for sig, data in blocks_struct_data.items():
            has_interfaces = len(data['sp_coord']) > 0
            has_orientations = len(data['op_coord']['Positions']) > 0
            
            # A block must have sufficient data to constrain a spatial drift trend
            if has_interfaces and has_orientations:
                print(f"\n--- Solving Structural Block {sig} ---")
                self.interface_data(data['sp_coord'])
                self.orientation_data(data['op_coord'])
                self.Transformation_matrix = structure_data.get('transformation_matrix', default_matrix)

                #Solving universal co-kriging for each block
                self.Ge_model()
                # Save the block's unique kriging weights associated with its signature
                self.block_states[sig] = self.save_internal_state()
            else:
                print(f"\nWarning: Block {sig} skipped (Needs at least 2 interface and 1 orientation point in this block).")


        # --- Combine Original Data for Visualization --- #
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
        Evaluates the final 3D geomodel by mapping the independently solved on grid using the boolean signature logic.
        """

        if grid_coord is None:
            grid_coord = self.data["Regular"]
        combined_backup = self.save_internal_state()

        try:
            grid_fault_masks = {}
            self.current_fault_z_dict = {}


            # Generate boolean masks for the entire grid based on fault topologies
            for f_name in self.fault_names:
                self.load_internal_state(self.fault_states[f_name])

                # Calculating scalar field on whole grid using current fault model weights
                out, _ = super().Solution_grid(grid_coord, section_plot=True, recompute_weights=False)
                
                self.current_fault_z_dict[f_name] = out["Regular"].clone()
                # Determine which grid points lie on which side of the current fault
                grid_fault_masks[f_name] = out["Regular"] > self.fault_thresholds[f_name]

           
            # Evaluate the continuous scalar fields for all structural blocks independently (ignoring faults for the moment).
            block_outputs, block_res = {}, {}
            for sig in self.block_states:
                self.load_internal_state(self.block_states[sig])
                out, res = super().Solution_grid(grid_coord, section_plot=True, recompute_weights=False)
                block_outputs[sig] = out
                block_res[sig] = res

            if not block_outputs:
                raise ValueError("No valid structural blocks were computed!")


            # Stitch blocks together using Boolean Signatures logic for plotting
            final_out, final_res = {}, {}
            template_sig = list(block_outputs.keys())[0]
           
            for k in block_outputs[template_sig].keys():
                final_out[k] = block_outputs[template_sig][k] if k == 'scalar_ref_points' else torch.zeros_like(block_outputs[template_sig][k])
            for k in block_res[template_sig].keys():
                final_res[k] = block_res[template_sig][k] if k == 'ref_points' else torch.zeros_like(block_res[template_sig][k])


            self.raw_block_scalars = {}
            self.raw_block_masks = {}
   
            # Apply fault masking
            for sig, out_dict in block_outputs.items():

                # Start with a mask of all True (entire grid)
                block_mask = torch.ones_like(grid_fault_masks[self.fault_names[0]], dtype=torch.bool)
                
                # Iteratively intersect the grid masks based on the block's Boolean Signature
                for i, f_name in enumerate(self.fault_names):
                    expected = sig[i]
                    if expected:
                        block_mask = block_mask & grid_fault_masks[f_name]
                    else:
                        block_mask = block_mask & (~grid_fault_masks[f_name])
            
                self.raw_block_masks[sig] = block_mask.clone()
                self.raw_block_scalars[sig] = out_dict['Regular'].clone()
    
                # Populate the final composite grid strictly where the block mask is True
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
        Function to stop new Solution_grid calculation inside the exisiting plotting methods of parent file.
        """
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
    fault0_trans = torch.diag(torch.tensor([1, 1, 1, 0.00], dtype=torch.float32))

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
            [2200.0, 500.0, 460.0, 0.0], ### This point was given bad results

            # [2300.0, 500.0, 460.0, 0.0],
            
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

            [-0.423, -0.000, 0.906, -0.007],

            [-0.423, -0.000, 0.906, -0.005]


        ])
    }
    
    struct_transformation_matrix = torch.diag(torch.tensor([1, 1, 1, 0.00], dtype=torch.float32))

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
    import time
    for t in [-0.5, 0, 0.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3,3.5, 4, 4.5]:
        model.plot_data_section(section={2:0.5, 4:t}, plot_scalar_field = True, plot_input_data=True)
        time.sleep(1)


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
    # model.plot_interactive_section(plot_input_data=True, only_surface_mode= False)