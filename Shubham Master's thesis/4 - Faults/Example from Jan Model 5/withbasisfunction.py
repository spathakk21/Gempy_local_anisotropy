import os
from functools import partial
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import pyvista as pv

# Integration of PyVista for 3D Visualization
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    print("Warning: PyVista not installed. Please install.")
    PYVISTA_AVAILABLE = False

import pyro
import pyro.distributions as dist


# for CI testing
smoke_test = ('CI' in os.environ)
pyro.set_rng_seed(1)


class grid:
    def __init__(self):
        super(grid, self).__init__()
        self.grid_data = []
        
    def Regular_grid(self, extent, resolution, dtype):
        
        dims = len(resolution)
        
        # Generate linspace per dimension
        for i in range(dims):
            start = extent[2*i]
            end = extent[2*i + 1]
            self.grid_data.append(torch.linspace(start, end, resolution[i], dtype=dtype))

        # Create meshgrid (cartesian product of all dimensions)
        self.mesh = torch.meshgrid(*self.grid_data, indexing="ij")  # Use 'ij' to keep dimensionality consistent

        # Flatten and stack each dimension
        flat = [m.reshape(-1) for m in self.mesh]
        full_grid = torch.stack(flat, dim=1)  # Shape: (num_points, num_dimensions)

        return full_grid
    
    def Custom_grid(self, custom_data):  
        return custom_data
        
        

    
class Gempy(grid):
    def __init__(self,project_name:str, extent:list, resolution:list):
        
        super(Gempy, self).__init__()
        self.project_name = project_name
        self.extent = extent
        self.resolution =resolution
        self.dtype = torch.float32
        self.grid_status={}
        self.sp_coord={}
        self.op_coord={}
        
        self.data={"Regular":None, "Custom":None}
        self.data["Regular"]= grid.Regular_grid(self, extent=self.extent, resolution=self.resolution, dtype=self.dtype)
        self.grid_status["Regular"]="Active"
        self.grid_status["Custom"] = None
        
        self.a_T = 5
        self.c_o_T = self.a_T**2/14/3
        self.s_1 = 0.01
        self.s_2 = 0.01
        self.Transformation_matrix = torch.eye(len(resolution), dtype=self.dtype)
    
        
            
    def interpolation_options(self):
        print("a_T =", self.a_T)
        print("c_o_T =", self.c_o_T)
        print("s_1 =", self.s_1)
        print("s_2 =",self.s_2)
        print("Transformation Matrix =\n", self.Transformation_matrix)
        
    def interpolation_options_set(self, a_T=5, c_o_T = 0.5952380952380952, s_1=0.01, s_2 = 0.01, Transformation_matrix=None):
        self.a_T = a_T
        self.c_o_T = c_o_T
        self.s_1 = s_1
        self.s_2 = s_2
        if Transformation_matrix is not None:
            self.Transformation_matrix = Transformation_matrix
            
        print("Interpolation option after modification: \n")
        self.interpolation_options()
        
    def interface_data(self, sp_coord):
       
        self.sp_coord = sp_coord
        print("\n ############# Interface data ############# \n",self.sp_coord)
        
        
    def orientation_data(self, op_coord):
        self.op_coord = op_coord
        print("\n ############# Orientation data############# \n",self.op_coord)
        
    def activate_custom_grid(self, custom_grid_data):
        if custom_grid_data.shape[1]==len(self.resolution):
            self.data["Custom"] = grid.Custom_grid(self, custom_grid_data)
            self.grid_status["Custom"]="Active"
            
                  
    def active_grid(self):
        print("Grid status\n", self.grid_status)
        
    def deactivate_grid(self, grid_type:str):
        self.grid_status[grid_type]= None
        print("Grid status\n", self.grid_status)
        
    def activate_grid(self, grid_type:str):
        self.grid_status[grid_type]= "Active"
        #print("Grid status\n", self.grid_status)
        
    
    
    def covariance_function(self, r):

        condition = r <=self.a_T

        r_by_at = r/self.a_T
        C_r = self.c_o_T *( 1 - 7 * (r_by_at)**2 + 8.75 * (r_by_at)**3 - 3.5 * (r_by_at)**5 + 0.75 * (r_by_at)**7)
        
        ## Because if r is much greater than the range a_T - the covariance function does not goes to zero
        C_r = torch.where(condition, C_r, 0.0)
        return C_r
    
    def first_derivative_covariance_function(self, r):

        condition = r <=self.a_T

        C_r_dash =self.c_o_T *( - 14 * (r/self.a_T**2) + 105/4 * (r**2/self.a_T**3) - 35/2 * (r**4/self.a_T**5) + 21/4 * (r**6/self.a_T**7))
        # C_r_dash =self.c_o_T *( - 14 * (r_by_at)**2 + 105/4 * (r_by_at)**3 - 35/2 * (r_by_at)**5 + 21/4 * (r_by_at)**7)/ r
        
        ##
        C_r_dash = torch.where(condition, C_r_dash, 0.0)
        
        return C_r_dash
    
    def first_derivative_covariance_function_divided_by_r(self, r):

        condition = r <=self.a_T

        C_r_dash_by_r = self.c_o_T *( - 14 / ((self.a_T)**2) + 105/4 * (r/(self.a_T)**3) - 35/2 * (r**3/(self.a_T)**5) + 21/4 * (r**5/(self.a_T)**7))
        
        ##
        C_r_dash_by_r = torch.where(condition, C_r_dash_by_r, 0.0)
        
        return C_r_dash_by_r
    
    def second_derivative_covariance_function(self,r):

        condition = r <=self.a_T

        C_r_dash_dash =self.c_o_T * 7 * (9 * r ** 5 - 20 * self.a_T ** 2 * r ** 3 + 15 * self.a_T ** 4 * r - 4 * self.a_T ** 5) / (2 * self.a_T ** 7)
        
        ##
        C_r_dash_dash = torch.where(condition, C_r_dash_dash, 0.0)

        
        return C_r_dash_dash

    def squared_euclidean_distance(self, x_1,x_2):
        x_1 = x_1 @ self.Transformation_matrix.T
        x_2 = x_2 @ self.Transformation_matrix.T

        sqd = torch.sqrt(torch.clip(torch.reshape(torch.sum(x_1**2,1),shape =(x_1.shape[0],1))+\
        torch.reshape(torch.sum(x_2**2,1),shape =(1,x_2.shape[0]))-\
        2*( x_1@ x_2.T), min=0.0)+ 1e-12)
        
        return sqd

    def cartesian_dist_hu(self, x1, x2 ):
        x1 = x1@ self.Transformation_matrix.T
        x2 = x2@ self.Transformation_matrix.T
        k = x1.shape[1]
        H =[]
        dummy_H = []
        # We converted x' = x A . 
        # There for delta x' = h_u (old) = delta x A
        for i in range(k):
            delta_x_i = x1[:,i] - torch.reshape(x2[:,i], shape=(x2.shape[0],1))
            H.append(delta_x_i)
        for i in range(k):
            a=torch.zeros(H[i].shape)
            for j in range(k):
                a= a +  self.Transformation_matrix[j,i] * H[j]
            dummy_H.append(a)
        H = dummy_H
        return H

    def cov_gradients(self, dist_tiled, H,  nugget_effect_grad=1/3):
        k = len(H) # component of gradient available
        n = int(H[0].shape[0]/ k) # number of place where gradient is defined

        #################################################################################################
        # For cross term of gradient 
        #################################################################################################
        # if we have many component of Gradient, we can club it to make , H =[h_u, h_v, h_w,....] 
        #H =[h_u, h_v] 
        C_G = torch.zeros((n*k, n*k))
        for i in range(len(H)):
            for j in range(len(H)):
                hu_hv = H[i][n*i:n*(i+1), n*j:n*(j+1)] * H[j][n*i:n*(i+1), n*j:n*(j+1)]
                dist =dist_tiled[n*i:n*(i+1), n*j:n*(j+1)]
                dist_sqr = dist **2 
                condition = dist_sqr!=0
                hu_hv_by_dist_sqr = torch.where(condition, hu_hv/ dist_sqr, 0.0)
                
                # Cross gradient term for C_ZuZv
                if i != j:
                    t2 = - self.first_derivative_covariance_function_divided_by_r(dist) + self.second_derivative_covariance_function(dist)
                    anisotrop_term = self.first_derivative_covariance_function_divided_by_r(dist) * torch.sum(self.Transformation_matrix[:,i]*self.Transformation_matrix[:,j])
                    array_test = -(hu_hv_by_dist_sqr * t2 + anisotrop_term)
                    for l in range(n):
                        for m in range(n):
                            C_G[i*n+l,j*n+m] = array_test[l,m]
                # Gradient for similar type of gradient
                else:
                    dist_cube = dist **3
                    condition = dist_cube!=0
                    hu_hv_by_dist_cube = torch.where(condition, hu_hv/dist_cube, 0.0)
                    
                    t4 = self.first_derivative_covariance_function(dist)
                    
                    # Here we have added the last term for anisotropy effect
                    t5 = torch.sum((self.Transformation_matrix[:,i])**2) * self.first_derivative_covariance_function_divided_by_r(dist) 
                    
                    t6 = self.second_derivative_covariance_function(dist)
                    
                    ################################# Note ############################################
                    # Covariance of gradient is negative for points closer to zero
                    ##################################################################################
            
                    array_test = -(- t4 * hu_hv_by_dist_cube + t5 +  t6 * hu_hv_by_dist_sqr )
                    for l in range(n):
                        for m in range(n):
                            C_G[i*n+l,j*n+m] = array_test[l,m]
        # If the distance is greater than a, then function C_z(r) = 0 . 
        # Therefore, we can replace the element of covariance matrix with 0 for distance greater than a
        condition_1 = dist_tiled<=self.a_T
        
        C_G = torch.where(condition_1, C_G, 0.0)
        C_G = C_G + nugget_effect_grad * torch.eye(n*k)

        return C_G

    def cov_interface(self, ref_layer_points,rest_layer_points, Transformation_matrix=torch.eye(2),nugget_effect_interface=1/3):
        sed_rest_rest = self.squared_euclidean_distance(rest_layer_points,rest_layer_points)
        sed_ref_rest = self.squared_euclidean_distance(ref_layer_points,rest_layer_points)
        sed_rest_ref = self.squared_euclidean_distance(rest_layer_points,ref_layer_points)
        sed_ref_ref = self.squared_euclidean_distance(ref_layer_points,ref_layer_points)
        
        C_I = self.covariance_function(sed_rest_rest) -\
            self.covariance_function(sed_ref_rest) -\
            self.covariance_function(sed_rest_ref) +\
            self.covariance_function(sed_ref_ref)

        return C_I + nugget_effect_interface * torch.eye(C_I.shape[0])

    ## Cartesian distance between dips and interface points

    def cartesian_dist_no_tile(self, x_1,x_2):
        x_1 = x_1 @ self.Transformation_matrix.T
        x_2 = x_2 @ self.Transformation_matrix.T
        k = x_1[0].shape[0]
        H_I = []
        for i in range(k):
            H_I.append(((x_1[:,i] - torch.reshape(x_2[:,i],[x_2.shape[0],1]))).T)
        
        Dummy_H_I=[]
        
        for i in range(k):
            a=torch.zeros(H_I[i].shape)
            for j in range(k):
                a= a + self.Transformation_matrix[j,i] * H_I[j]
            Dummy_H_I.append(a)
        
        H_I = Dummy_H_I
        H_I = torch.concat(H_I,axis=0)
        
        return H_I

    def cov_interface_gradients(self, hu_rest,hu_ref, Position_G_Modified, rest_layer_points, ref_layer_points):
        sed_dips_rest = self.squared_euclidean_distance(Position_G_Modified,rest_layer_points)
        sed_dips_ref = self.squared_euclidean_distance(Position_G_Modified,ref_layer_points)
        
        C_GI = - hu_rest * self.first_derivative_covariance_function_divided_by_r(sed_dips_rest) + hu_ref* self.first_derivative_covariance_function_divided_by_r(sed_dips_ref)
        
        return C_GI

    def set_rest_ref_matrix2(self, number_of_points_per_surface,input_position):
        ref_layer_points=[]
        rest_layer_points=[]
        for layer in input_position:
            ref_layer_points.append(layer[-1])
            rest_layer_points.append(layer[0:-1])
        # refrence points for each layer is repeated as number of non-referenced point
        
        repeats = number_of_points_per_surface-1

        ref_layer_points = torch.repeat_interleave(torch.stack(ref_layer_points ,axis = 0),repeats=repeats,axis = 0) 
        
        # Non referenced point. 
        rest_layer_points = torch.concat(rest_layer_points,axis = 0)
        return ref_layer_points,rest_layer_points
        


    ##############################################################################
    ########### SETTING BASIS FUNCTION FOR ADDING TREND TO THE MEAN ###############
    ##############################################################################
    def set_evaluate_basis(self, x, active_dims=[0, 1, 2, 3]):
        """
        Calculates the F matrix -- which represents polynomial trend of mean at data points
        Args:
            x: Tensor(N, dim) --> Input coordinates
            powers: List --> polynomial powers to calculate
            active_dims: List --> which dimensions to use. For eg. [0,1,2] means [X,Y,Z] and not time
            Takes all by default if none is given

        Returns:
            F_vals: (L, N) --> Value of basis functions at points
            F_grads: (L, N, dim) --> The derivatives of basis functions
        """

        # Get dimensions of input points
        N, dim = x.shape
        # N is the number of data points
        # dim is equal to 4
        # print(dim)
        
        # Default to all dimensions if not specified
        if active_dims is None:
            active_dims = list(range(dim))
        
        # Empty lists for storing values
        vals_list = []
        grads_list = []


        # 1. Linear Terms (x, y, z, t)
        for i in active_dims:
            # Value: x_i
            val = x[:, i]
            
            # Gradient: 1.0 at index i
            grad = torch.zeros(N, dim, dtype=self.dtype)
            
            # derivate of a variable with itself will be 1
            # so ith column is 1.0
            grad[:, i] = 1.0
            
            vals_list.append(val)
            grads_list.append(grad)

        # 2. Quadratic Terms (x^2, y^2, z^2, t^2)
        for i in active_dims:
            # Value: x_i^2
            val = x[:, i]**2
            
            # Power rule -- Gradient: 2 * x_i at index i
            grad = torch.zeros(N, dim, dtype=self.dtype)
            grad[:, i] = 2.0 * x[:, i]
            
            vals_list.append(val)
            grads_list.append(grad)

        # 3. Cross Terms (xy, xz, xt, yz, yt, zt)
        # Finding every unique pair from given pair of dimensions
        from itertools import combinations
        for i, j in combinations(active_dims, 2):
            #Multiplying coordinates
            val = x[:, i] * x[:, j]
            
            # Gradient:
            grad = torch.zeros(N, dim, dtype=self.dtype)
            grad[:, i] = x[:, j]
            grad[:, j] = x[:, i]
            
            vals_list.append(val)
            grads_list.append(grad)

        # Stack Results
        
        F_vals = torch.stack(vals_list, dim=0)  # Shape: (N_terms, N)
        F_grads = torch.stack(grads_list, dim=0) # Shape: (N_terms, N, dim)

        return F_vals, F_grads


    def Ge_model(self, nugget_effect_grad=1e-8,nugget_effect_interface=1e-8):
        ''' 
            Args:
                Input data:
                    input_position:     A list of list. Where each element of outer list is the information for each layer. 
                                        Inside each layer is the coordinates of position with last element is the refrence of that layer.
                    gradient_position:  A list of coordinates of gradient position
                    gradient_value:     Values of Gradient corresponding to the position mentioned in gradient_position
        '''
        # Assuming the there exist different layers in geological model. It can be assumed as scalar field. 
        
        self.number_of_layer = len(self.sp_coord)
        # Each layer containst the information of location of point where we have the information about scalar data. 
        # The last location is of the reference point for each layer.
        
        ## defining the dips position
        self.Position_G = self.op_coord["Positions"].to(self.dtype) # Location where Dips or gradient are given
        self.Value_G    = self.op_coord["Values"].to(self.dtype)     # @ self.Transformation_matrix.T # Gx, Gy, ..., Gk are the componet of gradient available at the given location
       
        n= self.Position_G.shape[0] # Total number of points available for gradient or dips
        k = self.Position_G[0].shape[0] # Total number of component available for the gradient
        # Since we have two component of the gradient, we can write the position twice corresponding to each coponent. We are assuming that 
        # Z is a scalar not a vector. Therefor we divide gradient into two different Z_u and Z_v
        
        Position_Add=[]
        for i in range(k):
            Position_Add.append(self.Position_G)
        self.Position_G_Modified = torch.cat(Position_Add, axis=0)
        
        number_of_points_per_surface=[]
        input_position=[]
        for keys, values in self.sp_coord.items():
            number_of_points_per_surface.append(len(values))
            input_position.append(values.to(self.dtype))

        self.number_of_points_per_surface = torch.tensor(number_of_points_per_surface)
        
        
        self.ref_layer_points,self.rest_layer_points = self.set_rest_ref_matrix2(self.number_of_points_per_surface,input_position)
        
        dist_position = self.squared_euclidean_distance(self.Position_G_Modified, self.Position_G_Modified) 
        # the dist_position can be all zero if there is only one point is defined where gradient is known.
        # 
        dist_position = dist_position #+  torch.eye(dist_position.shape[0])

        # Calculate the cartesian distance 

        H = self.cartesian_dist_hu(self.Position_G_Modified, self.Position_G_Modified)
        
        C_G = self.cov_gradients(dist_tiled=dist_position, H=H,nugget_effect_grad=nugget_effect_grad)
        
        C_I = self.cov_interface(self.ref_layer_points,self.rest_layer_points, nugget_effect_interface=nugget_effect_interface)
        
        hu_rest = self.cartesian_dist_no_tile(self.Position_G,self.rest_layer_points)
        hu_ref = self.cartesian_dist_no_tile(self.Position_G,self.ref_layer_points)   

        C_GI = self.cov_interface_gradients(hu_rest,hu_ref, self.Position_G_Modified, self.rest_layer_points, self.ref_layer_points)

        C_IG = C_GI.T

        K = torch.concat([torch.concat([C_G,C_GI],axis = 1),
        torch.concat([C_IG,C_I],axis = 1)],axis = 0)  

        # For kriging system in dual form require the list of all the Z. For gradient part, we can write the term but for Z(x)-Z(x0)=0 always.

        Modified_Value_G = self.Value_G #@ torch.eye(2)
        Modified_Value_G_flatten = torch.reshape((Modified_Value_G).T, [-1])
        b = torch.concat([Modified_Value_G_flatten,torch.zeros(K.shape[0]-Modified_Value_G_flatten.shape[0])],axis = 0)
        
        b = torch.reshape(b,shape = [b.shape[0],1])
        
        # ######################################
        # F MATRIX CONSTRUCTION
        # ######################################

        # Set on which variables I want basis to be applied
        # [0,1,2] means [x,y,z]
        active_dimensions = [0,1,2] 

        # Calculating Basis at Gradient Points
        _, dF_grad = self.set_evaluate_basis(self.Position_G, active_dims=active_dimensions) 
        # print(dF_grad) # Try printing to check F_grad structure

        # Reshaping dF_grad matrix
        L = dF_grad.shape[0]  # Number of basis polynomial terms(9) = [x,y,z, x2, y2, z2, xy,xz, yz]
        # print(L)


        n_grad_pts= self.Position_G.shape[0] # Total number of points available for gradient 
        # print(n_grad_pts)
        
        
        k_dim = self.Position_G.shape[1] # Total number of component available for the gradient
        # print(k_dim)
      

        # Initialize matrix for storing F_grad
        F_gradients = torch.zeros(L, n_grad_pts * k_dim)
        
        # Storing Gradient terms
        for u in range(k_dim):
            start_col = u * n_grad_pts
            end_col = (u + 1) * n_grad_pts
            F_gradients[:, start_col:end_col] = dF_grad[:, :, u]

    

        ########### Calculating Basis function value at Interface Points ######################

        # Using (reference - rest) for all the layers
        F_vals_rest, _ = self.set_evaluate_basis(self.rest_layer_points,active_dims=active_dimensions)
        F_vals_ref, _ = self.set_evaluate_basis(self.ref_layer_points, active_dims=active_dimensions)
        F_interface = -F_vals_rest + F_vals_ref 

        
        # Finalizing F_matrix
        F_matrix = torch.cat([F_gradients, F_interface], dim=1)
        self.F_matrix = F_matrix 
        
        # print(f"The F matrix is:{F_matrix}")

        # print(f"The shape of F matrix is:{F_matrix.shape}")

        
        # Augment K for solving
        ###### [C  F']
        ###### [F  0 ]

        zeros_corner = torch.zeros(L, L)
        top = torch.cat([K, F_matrix.T], dim=1)
        bottom = torch.cat([F_matrix, zeros_corner], dim=1)
        K_aug = torch.cat([top, bottom], dim=0)
        
        # RHS of equations
        # DOUBT
        zeros_b = torch.zeros(L, 1)
        b_aug = torch.cat([b, zeros_b], dim=0)
        
        # print(f"The RHS b vector is: {b_aug}")

        # Solve
        # w contains --> weights Covariance term(gradients and interface) and coefficients of basis term
        # TO DO --> check it
        self.w = torch.linalg.solve(K_aug, b_aug)
        
        # Storing these settings for Solution_grid
        self.num_basis_terms = L
        # self.last_powers = basis_powers
        self.last_active_dims = active_dimensions

    
    def Solution_grid(self, grid_coord, section_plot= False, recompute_weights=True):
        
        # INPUT --> grid_coord: grid points to evaluate

        # Optimization: Only solve the linear system if requested or if weights don't exist
        if recompute_weights or not hasattr(self, 'w'):
            self.Ge_model()
        
        self.ref_points = torch.unique(self.ref_layer_points,dim=0)
        #print("grid_coord shape",grid_coord.shape)
        if grid_coord is not None:
            grid_data_plus_ref = torch.concat([grid_coord, self.ref_points],dim=0)
        else:
            grid_data_plus_ref = self.ref_points
            
        ##############################################

        # Separating Weights into:
        # w_dual (gradient and interface) Covariance weights
        # mu = coefficients of basis polynomial for given dataset
        w_dual = self.w[:-self.num_basis_terms]
        # print(f"All wegihts excluding basis coefficients are: {w_dual}")
        # print(w_dual.shape)


        mu = self.w[-self.num_basis_terms:]
        # print(f"Basis coefficients are: {mu}")
        # print(mu.shape)

        # print(mu)

        ##############################################


        #print("grid_coord_plus shape",grid_data_plus_ref.shape)
        hu_Simpoints = self.cartesian_dist_no_tile(self.Position_G,grid_data_plus_ref)
        
        sed_dips_SimPoint = self.squared_euclidean_distance(self.Position_G_Modified,grid_data_plus_ref)
        

        ####################################### TODO #######################################
        # Check whether we need to transform first_derivative_covariance_function_divided_by_r 
        # by transformation matrix somehow
        ####################################################################################
        # Contribution from gradient data points
        
        # sigma_0_grad  =  self.w[:self.Position_G.shape[0] *self.Position_G.shape[1]] * (hu_Simpoints * self.first_derivative_covariance_function_divided_by_r(sed_dips_SimPoint))
        
        sigma_0_grad  =  w_dual[:self.Position_G.shape[0] *self.Position_G.shape[1]] * (hu_Simpoints * self.first_derivative_covariance_function_divided_by_r(sed_dips_SimPoint))

        # print(self.Position_G.shape[0] *self.Position_G.shape[1])

        sigma_0_grad = torch.sum(sigma_0_grad,axis=0)
        

        sed_rest_SimPoint = self.squared_euclidean_distance(self.rest_layer_points,grid_data_plus_ref)
        sed_ref_SimPoint = self.squared_euclidean_distance(self.ref_layer_points,grid_data_plus_ref)

        # Contribution from interface data points

        # sigma_0_interf =  self.w[self.Position_G.shape[0]*self.Position_G.shape[1]:]*(-self.covariance_function(sed_rest_SimPoint) + self.covariance_function(sed_ref_SimPoint))
        
        
        sigma_0_interf =  w_dual[self.Position_G.shape[0]*self.Position_G.shape[1]:]*(-self.covariance_function(sed_rest_SimPoint) + self.covariance_function(sed_ref_SimPoint))
        sigma_0_interf = torch.sum(sigma_0_interf,axis = 0)

        # print(f"self.covariance_function(sed_rest_SimPoint):{self.covariance_function(sed_rest_SimPoint)}")
        # print(f"self.covariance_function(sed_ref_SimPoint): {self.covariance_function(sed_ref_SimPoint)}")
        
        
        # print(self.Position_G.shape[0]*self.Position_G.shape[1])
        ################################################################

        ###########  Universal Calculation #############
        # Use the same settings (powers/dims) as used in Ge_model
        dims = getattr(self, 'last_active_dims', [0, 1, 2])
        
        ### Re-evaluate basis at the grid locations
        F_vals_sim, _ = self.set_evaluate_basis(grid_data_plus_ref, active_dims=dims)

        #### Coefficienets* Basis value <--> mu * f(x_grid)
        basis_value = torch.matmul(mu.T, F_vals_sim).squeeze()

        # Final estimation of Z*
        ### Z* = Covraince parts (grad + interface) + Basis function part
        interpolate_result = sigma_0_grad + sigma_0_interf  + basis_value

        # print(f"sigma_0_grad: {sigma_0_grad}")
        # print(f"sigma_0_interf: {sigma_0_interf}")
        # print(f"basis_value: {basis_value}")
        # print(f"interpolate_result: {interpolate_result}")

        ################################################################


        # interpolate_result = sigma_0_grad+ sigma_0_interf
        #print("interpolate_result ", interpolate_result.shape)
        scalar_field={}
        scalar_field["scalar_ref_points"] = interpolate_result[-self.number_of_layer:]
        if section_plot == False:
            start = 0
            for keys, values in self.grid_status.items():
                if values is not None:
                    end = start + self.data[keys].shape[0] ####
                    scalar_field[keys] = interpolate_result[start:end]
                    start = end
        else:
            scalar_field["Regular"] = interpolate_result[:-self.number_of_layer]
        
        #print(scalar_field["Regular"].shape)
        
        # scalar_ref_points = interpolate_result[-self.number_of_layer:]
        # interpolate_result_grid =interpolate_result[:-self.number_of_layer]
        
        # labels = torch.ones(interpolate_result_grid.shape[0])
        # i=1
        # label_index=[]
        # for ele in torch.sort(scalar_ref_points)[0]:
        #     label_index.append(i)
        #     mask = interpolate_result_grid >= ele
        #     labels= torch.where(mask, i+1, labels)
        #     i=i+1
            
    
        sorted_ref = torch.cat((interpolate_result.min().unsqueeze(0),torch.sort(scalar_field["scalar_ref_points"])[0], interpolate_result.max().unsqueeze(0)))
    
        #print(sorted_ref)
        
        modified_interpolate_results_final = torch.zeros(interpolate_result.shape)

        for i in range(len(sorted_ref)-2):
            modified_interpolate_results = torch.zeros(interpolate_result.shape)
            
            mask = (interpolate_result >= sorted_ref[i+1]-self.s_1) & (interpolate_result < sorted_ref[i+1]+ self.s_2)
            
            modified_interpolate_results =torch.where(mask, (interpolate_result - sorted_ref[i+1]+self.s_1)/(self.s_1 + self.s_2) , modified_interpolate_results)
            mask =  interpolate_result >  sorted_ref[i+1]+ self.s_2
            modified_interpolate_results =torch.where(mask, 1 , modified_interpolate_results)
            
            modified_interpolate_results_final +=  modified_interpolate_results
            
        modified_interpolate_results_final += 1
        results={}
        results["ref_points"] = modified_interpolate_results_final[-self.number_of_layer:]
        if section_plot == False:
            start = 0
            for keys, values in self.grid_status.items():
                if values is not None:
                    end = start + self.data[keys].shape[0]
                    results[keys] = modified_interpolate_results_final[start:end]
                    start = end
        else:
            results["Regular"] = modified_interpolate_results_final[:-self.number_of_layer]
                
        return scalar_field, results
        
    def Solution(self ):
        
        grid_data_final =[]
        
        for keys, values in self.grid_status.items():
            if values is not None:
               grid_data_final.append(self.data[keys]) 
                 
        if len(grid_data_final)>1:
            grid_data_ = torch.cat(grid_data_final, dim=0)
        elif len(grid_data_final)==1:
                grid_data_ = grid_data_final[0]
        else: 
            print("No grid is active")
            grid_data_ = None
            
        
        self.scalar_field, self.results = self.Solution_grid(grid_coord=grid_data_)   
        self.solution={}
        self.solution["scalar_field"]= self.scalar_field
        self.solution["result"]=self.results
        return self.solution
    
    def plot_2D(self, data ,sclar_field,  value, plot_scalar_field = True, plot_input_data=True,section=None):
        
        import matplotlib.pyplot as plt
        scatter =plt.scatter(data[:,0], data[:,1], c=value, cmap='viridis', s=100)
               
        axis_label = ["X", "Y", "Z", "T"]
        
        if section is None:
            accepted_index = [0,1]
        else:
            # Get the key to remove (convert to 0-based index)
            remove_index = list(section.keys())[0] - 1

            # Create a new list excluding that index
            accepted_index = [i for i, _ in enumerate(axis_label) if i != remove_index]
            
        if plot_scalar_field:
            X = self.mesh[0].numpy()
            Y = self.mesh[1].numpy()
            Z = sclar_field.reshape(X.shape).numpy()
            plt.contour(X, Y, Z)
        
        # Create a legend
        
        import numpy as np
        legend_labels = np.unique(value.numpy())  # Extract unique labels
        label_map={}
        label_map[1] = "Basement"
        i=1
        for keys, _ in self.sp_coord.items():
            label_map[i+1] = keys
            i = i+1 
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10, label=label_map[label]) for label in legend_labels]
        plt.legend(handles=legend_handles, title='Layers')
        plt.xlabel(axis_label[accepted_index[0]] + " Coordinates")
        plt.ylabel(axis_label[accepted_index[1]] + " Coordinates")
        plt.title('Scatter Plot with Color Labels')
        
        ########################################################################################
        ##### Plot surface points and gradients
        ########################################################################################
        if plot_input_data:
            colour = ['ro', 'bo', 'go']
            i=0
            for _, values in self.sp_coord.items():
                plt.plot(values[:,accepted_index[0]], values[:,accepted_index[1]], colour[i])
                i=i+1
            
            for i in range(self.Position_G.shape[0]):
                plt.plot(self.Position_G[i,accepted_index[0]], self.Position_G[i,accepted_index[1]], 'go')
                plt.quiver([self.Position_G[i,accepted_index[0]]],[self.Position_G[i,accepted_index[1]]],self.Value_G[i][accepted_index[0]],self.Value_G[i][accepted_index[1]],color='r')
        plt.savefig("Plot_2D_basis.png")
        plt.close()

        # ----
        # plt.show()


    def plot_3D(self, data ,  value, plot_scalar_field = True, plot_input_data=True, section=None):
        
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
 

        fig = plt.figure()

        
        ax = fig.add_subplot(111, projection='3d')

        points = data.numpy()  # shape: (N, 3)
        values = value.numpy()  # shape: (N,)

        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=values, cmap='viridis', s=0.1, alpha=0.5)
        # Create divider and append colorbar axis
    
        # Optional colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Label")
        import numpy as np
        legend_labels = np.unique(value.numpy())  # Extract unique labels
        label_map={}
        label_map[1] = "Basement"
        i=1
        for keys, _ in self.sp_coord.items():
            label_map[i+1] = keys
            i = i+1 
        
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10, label=label_map[label]) for label in legend_labels]
        plt.legend(handles=legend_handles, title='Layers',loc="upper left")
        
        ################################################################################
        #                                     TODO                                     #
        ################################################################################
        axis_label = ["X", "Y", "Z", "T"]
        
        if section is None:
            accepted_index = [0,1,2]
        else:
            # Get the key to remove (convert to 0-based index)
            remove_index = list(section.keys())[0] - 1

            # Create a new list excluding that index
            accepted_index = [i for i, _ in enumerate(axis_label) if i != remove_index]

            
        ax.set_xlabel(axis_label[accepted_index[0]])
        ax.set_ylabel(axis_label[accepted_index[1]])
        ax.set_zlabel(axis_label[accepted_index[2]], labelpad=-5)
            
        ax.set_title("3D Scatter Plot with Rounded Labels")
        
        
        ########################################################################################
        ##### Plot surface points and gradients
        ########################################################################################
        if plot_input_data:
            
            colour = ['ro', 'bo', 'go']
            i=0
            for _, values in self.sp_coord.items():
                ax.plot(values[:,accepted_index[0]], values[:,accepted_index[1]], values[:,accepted_index[2]], colour[i])
                
                # ---
                # ax.zaxis.set_inverted(True)
                i=i+1
            
            for i in range(self.Position_G.shape[0]):
                ax.plot(self.Position_G[i,accepted_index[0]], self.Position_G[i,accepted_index[1]],self.Position_G[i,accepted_index[2]], 'go')
                ax.quiver([self.Position_G[i,accepted_index[0]]],[self.Position_G[i,accepted_index[1]]],[self.Position_G[i,accepted_index[2]]],self.Value_G[i][accepted_index[0]],self.Value_G[i][accepted_index[1]],self.Value_G[i][accepted_index[2]],color='r')

        
        plt.savefig("Plot_3D_basis.png")
        plt.close()

        # ---
        
        # plt.draw()
        # plt.pause(0.05)    
   
    def plot_data(self, sol, plot_scalar_field = True, plot_input_data=True):
        
        if self.grid_status["Regular"] is not None:
            
            print("Regular grid is activate")
            
            self.dim = self.data["Regular"].shape[1]
            
            if self.dim == 2:
                print("Data has dimension 2 ")
                self.plot_2D(data=self.data["Regular"], sclar_field=sol["scalar_field"]["Regular"], value= torch.round(sol["result"]['Regular']), plot_scalar_field=plot_scalar_field, plot_input_data=plot_input_data)
                
                
            if self.dim==3:
                self.plot_3D(data=self.data["Regular"], value= torch.round(sol["result"]['Regular']), plot_scalar_field=plot_scalar_field, plot_input_data=plot_input_data)
                
            else:
                print("Since it is difficult to plot more than 3D for visualisation, provide the section along which you want to visualize to make it 2D or 3D plot")
                

    def plot_data_section(self, section, plot_scalar_field = True, plot_input_data=True):
        
        if self.grid_status["Regular"] is not None:
            
            print("Regular grid is activate")
            
            self.dim = self.data["Regular"].shape[1]
            
            if section is not None: 
                plot_dim = self.dim - len(section)
                print("Dimension of the plot would be : ", plot_dim)
                if plot_dim not in [2,3]:
                    print("Plot Dimesnion must be 2 or 3")
                    exit()
                else:
                    self.grid_hyperplane = []

                    # Generate linspace per dimension
                    for i in range(len(self.resolution)):
                        if i+1 in section:
                            self.grid_hyperplane.append(torch.tensor(section[i+1], dtype=self.dtype))
                        else:
                            start = self.extent[2*i]
                            end = self.extent[2*i + 1]
                            self.grid_hyperplane.append(torch.linspace(start, end, self.resolution[i], dtype=self.dtype))
                    # Create meshgrid (cartesian product of all dimensions)
                    mesh = torch.meshgrid(*self.grid_hyperplane, indexing="ij")  # Use 'ij' to keep dimensionality consistent
                    
                    # Flatten and stack each dimension
                    flat = [m.reshape(-1) for m in mesh]
                    full_grid_hyp = torch.stack(flat, dim=1)  # Shape: (num_points, num_dimensions)

                    # Find the solution at this location
                    scalar_field, results = self.Solution_grid(grid_coord=full_grid_hyp,section_plot=True)
                    
                    # Plot
                    columns_to_keep = [i for i in range(full_grid_hyp.shape[1]) if i + 1 not in section]
                    final_grid = full_grid_hyp[:, columns_to_keep]
                    
                    mesh_selected = tuple(mesh[i].squeeze() for i in columns_to_keep)
                    self.mesh = mesh_selected
                    if plot_dim==2:
                        #print(final_grid.shape, scalar_field["Regular"] )
                        
                        
                        self.plot_2D(data=final_grid, sclar_field=scalar_field["Regular"], value= torch.round(results['Regular']), plot_scalar_field=plot_scalar_field, plot_input_data=plot_input_data, section=section)
                    else:
                        self.plot_3D(data=final_grid, value= torch.round(results['Regular']), plot_scalar_field=plot_scalar_field, plot_input_data=plot_input_data,section=section) 
                        
    
                
            else:
                print("Provide a valid section")
    
    ########## ---            
    def get_section_grid(self, section):
        """Helper to generate grid points for a slice."""
        grid_hyperplane = []
        # Generate linspace per dimension
        for i in range(len(self.resolution)):
            if i+1 in section:
                grid_hyperplane.append(torch.tensor(section[i+1], dtype=self.dtype))
            else:
                start = self.extent[2*i]
                end = self.extent[2*i + 1]
                grid_hyperplane.append(torch.linspace(start, end, self.resolution[i], dtype=self.dtype))
        
        # Create meshgrid
        mesh = torch.meshgrid(*grid_hyperplane, indexing="ij")  
        flat = [m.reshape(-1) for m in mesh]
        full_grid_hyp = torch.stack(flat, dim=1)
        
        # Determine which columns to keep for the plotting coordinates
        columns_to_keep = [i for i in range(full_grid_hyp.shape[1]) if i + 1 not in section]
        final_grid_coords = full_grid_hyp[:, columns_to_keep]
        
        return full_grid_hyp, final_grid_coords

   
    ##### ---
    def plot_interactive_section(self, plot_input_data=True, only_surface_mode = False):
        """
        Creates an interactive 3D PyVista plot with a time slider 
        'Args:
        plot_input_data: if you want to view the interface points and gradient vectors - set it to True
         
        only_surface_mode: if you want to view only the contoured surface with the input points, no point cloud - set it to True
        
        """
        if not PYVISTA_AVAILABLE:
            print("PyVista is required for interactive plotting.")
            return

        slider_dim=4
        # Setup range for the slider dimension
        # Convert 1-based index to 0-based
        idx = slider_dim - 1
        t_min = self.extent[2*idx]   # taking t value directly from model extent definition
        t_max = self.extent[2*idx + 1]
        
        # Initial Plot at min value
        current_t = t_min
        current_section = {slider_dim : current_t}
        
        # Generates the grid points for the initial frame
        full_grid_hyp, final_grid = self.get_section_grid(current_section)
        
        ##Pre-compute weights if missing ##
        # This ensures the slider is fast because we don't invert the matrix every frame
        if not hasattr(self, 'w'):
            self.Ge_model() 

        # Compute initial solution
        scalar_field, results = self.Solution_grid(grid_coord=full_grid_hyp, section_plot=True, recompute_weights=False)
        
        # print(scalar_field, results)

        # Determines location - grid points
        points = final_grid.numpy()

        # Determines colour
        # Rounds float results to integers (Rock IDs)
        values = torch.round(results['Regular']).numpy()
        
        # Create PyVista Mesh - point cloud
        mesh = pv.PolyData(points)
        # attaching rock id to mesh
        mesh["Lithology"] = values
        
        # Visualization window
        plotter = pv.Plotter(window_size=[1024, 768])


        if only_surface_mode is False:

            plotter.add_mesh(mesh, scalars="Lithology", cmap="viridis", 
                         point_size=5, render_points_as_spheres=True, opacity=0.5,
                         show_scalar_bar=False,label="Geological Grid")

        # #########################
        # ADDING LEGEND 
        # ########################
        label_map = {1: "Basement"}
        i = 1
        for key in self.sp_coord.keys():
            label_map[i+1] = key
            i += 1
            
        cmap = plt.get_cmap("viridis")
        
        # Number of rock types = Basement + defined layers
        max_possible_val = 1 + len(self.sp_coord)
        # Creates a scaling function. It maps IDs (1 to 2) to the color range (0.0 to 1.0).
        norm = plt.Normalize(vmin=1, vmax=max_possible_val)
        

        # Actual pair list (Name, Colour) for legend box
        legend_entries = []
        # Add all possible layers to legend, not just currently visible ones, so legend is stable
        for val in range(1, max_possible_val + 1):
             color = cmap(norm(val))
             name = label_map.get(val, f"Layer {val}")
             legend_entries.append((name, color))
             
        plotter.add_legend(legend_entries, loc = 'lower right')


        # Add XYZ coordinate arrows 
        plotter.add_axes()

        # Show X,Y,Z extent on the grid box
        plotter.show_grid()


        ####################################################################
        ################ For Contouring the interface ######################
        ####################################################################


        # using pyvista structure grid - to get connected gird

        # Grid dimensions from resolution of Gempy model
        nx, ny, nz = self.resolution[0], self.resolution[1], self.resolution[2]
        
        vol_grid = pv.StructuredGrid()
        vol_grid.points = points
        vol_grid.dimensions = [nx, ny, nz] 

        # Assign the rock IDs to the every point in volume
        vol_grid["Lithology"] = values


        # Generate Initial Contours
        # Taking surfaces at 1.5, 2.5, etc. (The boundary between ID 1 and 2)
        max_layer_val = values.max()
        contour_levels = [i + 0.5 for i in range(1, int(max_layer_val) + 1)]
        #####  contour_levels = [1.5, 2.5, 3.5, ......]

        # Calculating contour surface between two rocks 
        # interfaces is the new mesh object
        interfaces = vol_grid.contour(isosurfaces=contour_levels)

        # Adding contours to the plotter
        interface_actor = plotter.add_mesh(interfaces, color="white", opacity=0.7, label="Interface")

        ##########################################################################
                        # PLOTTING INPUT DATA #
        ##########################################################################

        if plot_input_data:
            input_colours = ["red", "blue", "green"]

            i = 0
            for _, coords in self.sp_coord.items():
                if coords.shape[1] > 3:
                     # Remove 4th dimension (Time) for plotting location
                    valid_coords = coords[:,[0,1,2]].numpy()
                else:
                    valid_coords = coords.numpy()

                c = input_colours[i % len(input_colours)]
                
                plotter.add_points(valid_coords, color=c, point_size=12, 
                                   render_points_as_spheres=True, label=f"Input: {key}")
                i += 1

            ######## ADDING GRADIENT ARROWS #########

            if hasattr(self, 'Position_G') and hasattr(self, 'Value_G'):
                # Extract XYZ for positions
                pos_g = self.Position_G[:, [0, 1, 2]].numpy() if self.Position_G.shape[1] > 3 else self.Position_G.numpy()
                
                # Extract XYZ for vectors (ignoring Time component of the vector if it exists)
                vec_g = self.Value_G[:, [0, 1, 2]].numpy() if self.Value_G.shape[1] > 3 else self.Value_G.numpy()
                
                # Create PyVista Arrow object
                arrows = pv.PolyData(pos_g)
                arrows["vectors"] = vec_g
                # "Glyph" filters scale geometry (arrows) at every point
                arrow_glyph = arrows.glyph(orient="vectors", scale=False, factor=0.4)
                
                plotter.add_mesh(arrow_glyph, color="red", label="Gradients")

    

        ###### Callback for time slider #######

        def on_slider_change(t_value):

            ''' Value is the new Time t.'''

            # old contour variable
            nonlocal interface_actor         #Allow updating the contour variable

            # Update section
            new_section = {slider_dim: t_value}
            new_section[slider_dim] = t_value
            

            # Recalculate Grid positions
            # Regenerates grid points for the new t
            new_full_grid, _ = self.get_section_grid(new_section)
            
            # Re-calculates lithology for new points and (Fast, weights are solved already)
            _, new_result = self.Solution_grid(grid_coord=new_full_grid, section_plot=True, recompute_weights=False)
            new_values = torch.round(new_result['Regular']).numpy()
            
            # Updates the colors on the existing mesh
            mesh["Lithology"] = torch.round(new_result['Regular']).numpy()


            # ###Update Contours ###


            # Update the volume grid data
            vol_grid.points = final_grid.numpy() 

            #  Updates the grid with the new Rock IDs calculated for the new Time (t)
            vol_grid["Lithology"] = new_values   
            
            # Recalculate the new contour surface
            new_interfaces = vol_grid.contour(isosurfaces=contour_levels)
            
            # Swap the actor in the scene - removes the old contour surface as we change t
            plotter.remove_actor(interface_actor)

            # checks if there are boundaries to draw or not
            if new_interfaces.n_points > 0:
                interface_actor = plotter.add_mesh(new_interfaces, color="white", opacity=0.7)


            return

        # Add Slider
        plotter.add_slider_widget(on_slider_change, [t_min, t_max], 
                                  title=f"T Evolution",
                                  pointa=(0.65, 0.90), # start of slider
                                  pointb=(0.95, 0.90), # end of slider
                                  color="black")
        
        print(f"Opening Interactive Plot. Use the slider to change T dimension...")
        plotter.show()

def main():        
    

   #### CHECKING jan_models dataset model5 (without fault)

    Transformation_matrix = torch.diag(torch.tensor([1,1,1,0.05],dtype=torch.float32))

    gp = Gempy("Strata_model", 
               extent = [-0.1, 1.1, 0.1, 0.9, 0.1, 0.90, 0, 5],
               resolution = [100, 50, 50, 2]
               )
    
   

    interface_data = {
    "rock1": torch.tensor([
        [0.0, 200.0, 600.0, 0.0],
        [0.0, 500.0, 600.0, 0.0],
        [0.0, 800.0, 600.0, 0.0],
        [200.0, 200.0, 600.0, 0.0],
        [200.0, 500.0, 600.0, 0.0],
        [200.0, 800.0, 600.0, 0.0],
        [800.0, 200.0, 200.0, 0.0],
        [800.0, 500.0, 200.0, 0.0],
        [800.0, 800.0, 200.0, 0.0],
        [1000.0, 200.0, 200.0, 0.0],
        [1000.0, 500.0, 200.0, 0.0],
        [1000.0, 800.0, 200.0, 0.0],
    ]) / 1000,

    "rock2": torch.tensor([
        [0.0, 200.0, 800.0, 0.0],
        [0.0, 800.0, 800.0, 0.0],
        [200.0, 200.0, 800.0, 0.0],
        [200.0, 800.0, 800.0, 0.0],
        [800.0, 200.0, 400.0, 0.0],
        [800.0, 800.0, 400.0, 0.0],
        [1000.0, 200.0, 400.0, 0.0],
        [1000.0, 800.0, 400.0, 0.0],
    ]) / 1000
    
}

    orientation_data = {
    "Positions": torch.tensor([
        [100.0, 500.0, 800.0, 0.0],
        [100.0, 500.0, 600.0, 0.0],
        [900.0, 500.0, 400.0, 0.0],
        [900.0, 500.0, 200.0, 0.0],
    ]) / 1000,

    "Values": torch.tensor([
        [0.000, 0.000, 1.000, 0.0],
        [0.000, 0.000, 1.000, 0.0],
        [0.000, 0.000, 1.000, 0.0],
        [0.000, 0.000, 1.000, 0.0],
    ])
}
    

#     # GULFAKS DATASET
#     interface_data = {

#  'etive': torch.tensor([[0.0624, 0.1384, 0.2240, 0.0000],
#          [0.1136, 0.0571, 0.3016, 0.0000],
#          [0.0217, 0.0427, 0.1651, 0.0000],
#          [0.0101, 0.2308, 0.1256, 0.0000],
#          [0.1169, 0.2148, 0.3174, 0.0000],
#          [0.0656, 0.2273, 0.2131, 0.0000],
#          [0.0423, 0.3136, 0.1666, 0.0000],
#          [0.0000, 0.4439, 0.0756, 0.0000],
#          [0.0894, 0.3537, 0.2461, 0.0000],
#          [0.0504, 0.3756, 0.1733, 0.0000],
#          [0.0787, 0.4736, 0.2016, 0.0000],
#          [0.0530, 0.5919, 0.1628, 0.0000],
#          [0.0240, 0.7489, 0.1160, 0.0000],
#          [0.1027, 0.6534, 0.2278, 0.0000],
#          [0.0677, 0.7011, 0.1906, 0.0000],
#          [0.0835, 0.7704, 0.1963, 0.0000],
#          [0.0569, 0.8164, 0.1551, 0.0000],
#          [0.1039, 0.8235, 0.2050, 0.0000],
#          [0.0686, 0.8875, 0.1478, 0.0000],
#          [0.0363, 0.9721, 0.0113, 0.0000],
#          [0.0167, 0.9124, 0.1154, 0.0000],
#          [0.1052, 0.9540, 0.1480, 0.0000],
#          [0.4014, 0.1267, 0.3372, 0.0000],
#          [0.3815, 0.0213, 0.3299, 0.0000],
#          [0.3268, 0.1097, 0.2438, 0.0000],
#          [0.4660, 0.0289, 0.4304, 0.0000],
#          [0.5969, 0.1145, 0.6141, 0.0000],
#          [0.6112, 0.0160, 0.6325, 0.0000],
#          [0.5682, 0.0447, 0.5560, 0.0000],
#          [0.5090, 0.1346, 0.4884, 0.0000],
#          [0.4585, 0.2359, 0.3930, 0.0000],
#          [0.5904, 0.1978, 0.6004, 0.0000],
#          [0.3084, 0.5356, 0.1794, 0.0000],
#          [0.3656, 0.3728, 0.2595, 0.0000],
#          [0.3153, 0.3101, 0.2094, 0.0000],
#          [0.3825, 0.2440, 0.2950, 0.0000],
#          [0.5160, 0.2583, 0.4744, 0.0000],
#          [0.5979, 0.2643, 0.6107, 0.0000],
#          [0.5543, 0.2887, 0.5347, 0.0000],
#          [0.4162, 0.3812, 0.3053, 0.0000],
#          [0.4774, 0.3400, 0.3919, 0.0000],
#          [0.5942, 0.3334, 0.6118, 0.0000],
#          [0.3963, 0.5102, 0.2674, 0.0000],
#          [0.5411, 0.4279, 0.4892, 0.0000],
#          [0.6343, 0.3718, 0.6879, 0.0000],
#          [0.6270, 0.4345, 0.6632, 0.0000],
#          [0.3427, 0.6336, 0.1898, 0.0000],
#          [0.3840, 0.6752, 0.2269, 0.0000],
#          [0.4589, 0.4469, 0.3397, 0.0000],
#          [0.6192, 0.5246, 0.6324, 0.0000],
#          [0.5689, 0.5178, 0.5201, 0.0000],
#          [0.3306, 0.7136, 0.1775, 0.0000],
#          [0.4692, 0.5419, 0.3519, 0.0000],
#          [0.5321, 0.5574, 0.4558, 0.0000],
#          [0.5777, 0.5899, 0.5391, 0.0000],
#          [0.5971, 0.6704, 0.5677, 0.0000],
#          [0.4323, 0.6336, 0.2810, 0.0000],
#          [0.5086, 0.6468, 0.4023, 0.0000],
#          [0.4551, 0.7308, 0.3123, 0.0000],
#          [0.5273, 0.7445, 0.4308, 0.0000],
#          [0.4237, 0.8068, 0.2723, 0.0000],
#          [0.3850, 0.9073, 0.2307, 0.0000],
#          [0.5904, 0.7651, 0.5508, 0.0000],
#          [0.5702, 0.8154, 0.4986, 0.0000],
#          [0.4631, 0.9906, 0.3010, 0.0000],
#          [0.5379, 0.9802, 0.4227, 0.0000],
#          [0.4681, 0.8626, 0.3377, 0.0000],
#          [0.3102, 0.7984, 0.1791, 0.0000],
#          [0.5423, 0.8718, 0.4475, 0.0000],
#          [0.5911, 0.8464, 0.5249, 0.0000],
#          [0.5938, 0.9086, 0.5347, 0.0000],
#          [0.6051, 0.9632, 0.5393, 0.0000],
#          [0.8766, 0.2181, 0.5177, 0.0000],
#          [0.9076, 0.2090, 0.5529, 0.0000],
#          [0.9019, 0.1244, 0.5116, 0.0000],
#          [0.9487, 0.1907, 0.6126, 0.0000],
#          [0.8905, 0.0335, 0.4699, 0.0000],
#          [0.8512, 0.0216, 0.5184, 0.0000],
#          [0.9719, 0.0988, 0.5646, 0.0000],
#          [0.9908, 0.1765, 0.7113, 0.0000],
#          [0.9508, 0.2745, 0.6223, 0.0000],
#          [0.9045, 0.2999, 0.5467, 0.0000],
#          [0.9878, 0.2707, 0.7025, 0.0000],
#          [0.8517, 0.3860, 0.5034, 0.0000],
#          [0.9373, 0.3903, 0.5926, 0.0000],
#          [0.9015, 0.4759, 0.5318, 0.0000],
#          [0.7792, 0.9782, 0.4726, 0.0000],
#          [0.8557, 0.5500, 0.5025, 0.0000],
#          [0.9775, 0.3725, 0.6560, 0.0000],
#          [0.9996, 0.4106, 0.6849, 0.0000],
#          [0.9869, 0.4540, 0.6628, 0.0000],
#          [0.8810, 0.6130, 0.5144, 0.0000],
#          [0.9353, 0.5124, 0.5630, 0.0000],
#          [0.9658, 0.5295, 0.6065, 0.0000],
#          [0.8137, 0.9670, 0.4828, 0.0000],
#          [0.9073, 0.7575, 0.5545, 0.0000],
#          [0.9150, 0.5432, 0.5406, 0.0000],
#          [0.9381, 0.6534, 0.5798, 0.0000],
#          [0.9334, 0.5818, 0.5495, 0.0000],
#          [0.9929, 0.6059, 0.6469, 0.0000],
#          [0.9976, 0.7725, 0.6366, 0.0000],
#          [0.8527, 1.0000, 0.5232, 0.0000],
#          [0.9769, 0.0262, 0.5539, 0.0000]]),

#           'ness': torch.tensor([[4.9428e-02, 1.5592e-01, 3.7559e-01, 0.0000e+00],
#          [6.0465e-02, 2.9457e-02, 4.0316e-01, 0.0000e+00],
#          [2.4787e-02, 2.9964e-02, 3.4357e-01, 0.0000e+00],
#          [8.6951e-02, 1.2671e-01, 4.4959e-01, 0.0000e+00],
#          [1.0688e-01, 2.9203e-02, 4.9824e-01, 0.0000e+00],
#          [1.8179e-02, 1.4347e-01, 3.2284e-01, 0.0000e+00],
#          [2.4191e-02, 2.9609e-01, 3.1268e-01, 0.0000e+00],
#          [7.9975e-02, 1.9731e-01, 4.0106e-01, 0.0000e+00],
#          [1.0685e-01, 2.3591e-01, 4.5871e-01, 0.0000e+00],
#          [6.0289e-02, 2.9812e-01, 3.6332e-01, 0.0000e+00],
#          [9.2018e-02, 2.8415e-01, 4.1913e-01, 0.0000e+00],
#          [5.9579e-02, 4.6064e-01, 3.2808e-01, 0.0000e+00],
#          [2.6742e-02, 4.4566e-01, 2.8715e-01, 0.0000e+00],
#          [8.1960e-02, 3.8573e-01, 3.8496e-01, 0.0000e+00],
#          [2.8966e-02, 6.2037e-01, 2.7302e-01, 0.0000e+00],
#          [9.3303e-02, 4.6800e-01, 3.9116e-01, 0.0000e+00],
#          [8.1504e-02, 5.3428e-01, 3.7562e-01, 0.0000e+00],
#          [6.6175e-02, 5.6729e-01, 3.3475e-01, 0.0000e+00],
#          [7.6930e-02, 6.5338e-01, 3.6950e-01, 0.0000e+00],
#          [8.2390e-02, 5.8507e-01, 3.9426e-01, 0.0000e+00],
#          [2.9374e-02, 7.8796e-01, 2.6859e-01, 0.0000e+00],
#          [7.4068e-02, 7.7628e-01, 3.6260e-01, 0.0000e+00],
#          [6.6840e-02, 8.6211e-01, 3.3115e-01, 0.0000e+00],
#          [9.5332e-02, 8.9512e-01, 3.8773e-01, 0.0000e+00],
#          [1.2370e-01, 9.4642e-01, 4.2131e-01, 0.0000e+00],
#          [7.4719e-02, 9.6521e-01, 3.3997e-01, 0.0000e+00],
#          [2.7038e-02, 9.6242e-01, 2.5603e-01, 0.0000e+00],
#          [1.6396e-01, 9.5328e-01, 4.8567e-01, 0.0000e+00],
#          [3.8257e-01, 1.3941e-01, 4.9086e-01, 0.0000e+00],
#          [4.1401e-01, 5.0787e-04, 5.4510e-01, 0.0000e+00],
#          [3.2761e-01, 0.0000e+00, 4.4040e-01, 0.0000e+00],
#          [3.1430e-01, 2.0899e-01, 4.1040e-01, 0.0000e+00],
#          [2.8065e-01, 4.1392e-02, 3.9919e-01, 0.0000e+00],
#          [5.0177e-01, 1.0284e-01, 6.5441e-01, 0.0000e+00],
#          [5.4363e-01, 2.6155e-02, 7.0384e-01, 0.0000e+00],
#          [4.8495e-01, 7.6181e-04, 6.3662e-01, 0.0000e+00],
#          [5.7197e-01, 1.5058e-01, 7.2844e-01, 0.0000e+00],
#          [5.9516e-01, 2.6155e-02, 7.6124e-01, 0.0000e+00],
#          [4.7200e-01, 2.5800e-01, 5.9678e-01, 0.0000e+00],
#          [5.4024e-01, 1.6480e-01, 6.9375e-01, 0.0000e+00],
#          [3.8307e-01, 2.9355e-01, 4.9133e-01, 0.0000e+00],
#          [5.9886e-01, 2.5927e-01, 7.7257e-01, 0.0000e+00],
#          [5.6691e-01, 2.3032e-01, 7.2024e-01, 0.0000e+00],
#          [2.3980e-01, 3.3900e-01, 3.4414e-01, 0.0000e+00],
#          [2.8467e-01, 4.3017e-01, 3.5860e-01, 0.0000e+00],
#          [3.3771e-01, 3.8674e-01, 4.3339e-01, 0.0000e+00],
#          [5.4016e-01, 2.9533e-01, 6.9859e-01, 0.0000e+00],
#          [5.7259e-01, 3.4561e-01, 7.5610e-01, 0.0000e+00],
#          [5.9712e-01, 3.6364e-01, 7.9356e-01, 0.0000e+00],
#          [3.2480e-01, 6.2189e-01, 3.6502e-01, 0.0000e+00],
#          [4.5579e-01, 4.2128e-01, 5.4663e-01, 0.0000e+00],
#          [5.2452e-01, 4.1899e-01, 6.5724e-01, 0.0000e+00],
#          [2.4886e-01, 5.3555e-01, 3.2908e-01, 0.0000e+00],
#          [3.9521e-01, 5.0406e-01, 4.6448e-01, 0.0000e+00],
#          [2.5763e-01, 6.0818e-01, 3.1089e-01, 0.0000e+00],
#          [2.7055e-01, 5.6450e-01, 3.3214e-01, 0.0000e+00],
#          [4.9274e-01, 5.7110e-01, 5.7216e-01, 0.0000e+00],
#          [5.6864e-01, 5.4850e-01, 7.2069e-01, 0.0000e+00],
#          [3.9311e-01, 6.6684e-01, 4.1076e-01, 0.0000e+00],
#          [2.5918e-01, 6.8690e-01, 3.1984e-01, 0.0000e+00],
#          [5.7594e-01, 6.6455e-01, 6.9153e-01, 0.0000e+00],
#          [5.4160e-01, 6.9705e-01, 6.0166e-01, 0.0000e+00],
#          [4.6383e-01, 7.4683e-01, 4.6990e-01, 0.0000e+00],
#          [2.7782e-01, 7.9787e-01, 3.4320e-01, 0.0000e+00],
#          [5.8115e-01, 8.1386e-01, 6.7032e-01, 0.0000e+00],
#          [3.1297e-01, 8.1361e-01, 3.4005e-01, 0.0000e+00],
#          [5.3507e-01, 8.4027e-01, 5.7363e-01, 0.0000e+00],
#          [3.7518e-01, 8.2707e-01, 3.6739e-01, 0.0000e+00],
#          [3.1707e-01, 9.8426e-01, 3.1906e-01, 0.0000e+00],
#          [4.7093e-01, 9.6953e-01, 4.8132e-01, 0.0000e+00],
#          [5.4101e-01, 9.6750e-01, 5.9648e-01, 0.0000e+00],
#          [3.9545e-01, 9.7791e-01, 3.9576e-01, 0.0000e+00],
#          [8.9297e-01, 3.3596e-01, 7.0472e-01, 0.0000e+00],
#          [8.8988e-01, 2.4200e-01, 7.0219e-01, 0.0000e+00],
#          [8.6383e-01, 2.0518e-01, 6.6923e-01, 0.0000e+00],
#          [9.7207e-01, 2.5063e-01, 8.3132e-01, 0.0000e+00],
#          [9.2662e-01, 1.9756e-01, 7.5965e-01, 0.0000e+00],
#          [8.1993e-01, 3.8116e-01, 6.2553e-01, 0.0000e+00],
#          [8.6330e-01, 4.2280e-01, 6.6972e-01, 0.0000e+00],
#          [7.7075e-01, 6.1783e-01, 5.7911e-01, 0.0000e+00],
#          [8.2963e-01, 4.6953e-01, 6.2085e-01, 0.0000e+00],
#          [9.3339e-01, 3.8268e-01, 7.5825e-01, 0.0000e+00],
#          [9.5605e-01, 4.6369e-01, 7.6193e-01, 0.0000e+00],
#          [9.8914e-01, 6.4525e-01, 7.6809e-01, 0.0000e+00],
#          [9.0498e-01, 4.3753e-01, 6.9977e-01, 0.0000e+00],
#          [9.2308e-01, 4.8908e-01, 7.0392e-01, 0.0000e+00],
#          [8.7796e-01, 5.0559e-01, 6.4769e-01, 0.0000e+00],
#          [8.3400e-01, 7.5013e-01, 6.4491e-01, 0.0000e+00],
#          [8.7180e-01, 6.5211e-01, 6.3752e-01, 0.0000e+00],
#          [8.3495e-01, 5.9142e-01, 6.0477e-01, 0.0000e+00],
#          [8.9111e-01, 5.7847e-01, 6.6794e-01, 0.0000e+00],
#          [9.3652e-01, 5.8583e-01, 7.1623e-01, 0.0000e+00],
#          [9.0919e-01, 7.0467e-01, 6.8790e-01, 0.0000e+00],
#          [7.9911e-01, 6.6557e-01, 5.8388e-01, 0.0000e+00],
#          [7.8685e-01, 8.2936e-01, 6.2191e-01, 0.0000e+00],
#          [7.3951e-01, 7.9228e-01, 5.2941e-01, 0.0000e+00],
#          [8.2650e-01, 8.5729e-01, 6.4636e-01, 0.0000e+00],
#          [8.8124e-01, 9.4820e-01, 6.8946e-01, 0.0000e+00],
#          [7.4970e-01, 9.2763e-01, 5.4299e-01, 0.0000e+00],
#          [8.5924e-01, 8.6059e-01, 6.6147e-01, 0.0000e+00],
#          [8.1398e-01, 9.8045e-01, 6.8034e-01, 0.0000e+00],
#          [7.8029e-01, 9.2153e-01, 6.2159e-01, 0.0000e+00]]),

#          'tarbert': torch.tensor([[0.0596, 0.1059, 0.5649, 0.0000],
#          [0.0416, 0.0122, 0.5624, 0.0000],
#          [0.0147, 0.1762, 0.4958, 0.0000],
#          [0.0888, 0.1470, 0.5937, 0.0000],
#          [0.0565, 0.2128, 0.5603, 0.0000],
#          [0.0828, 0.2781, 0.5779, 0.0000],
#          [0.0357, 0.2552, 0.5494, 0.0000],
#          [0.0505, 0.2938, 0.5431, 0.0000],
#          [0.0256, 0.3593, 0.5257, 0.0000],
#          [0.0550, 0.4975, 0.5553, 0.0000],
#          [0.0273, 0.4939, 0.5307, 0.0000],
#          [0.0526, 0.5688, 0.5426, 0.0000],
#          [0.0285, 0.6539, 0.5305, 0.0000],
#          [0.0564, 0.6437, 0.5659, 0.0000],
#          [0.0584, 0.7164, 0.5553, 0.0000],
#          [0.0311, 0.8131, 0.5213, 0.0000],
#          [0.0629, 0.8220, 0.5581, 0.0000],
#          [0.0749, 0.9652, 0.5551, 0.0000],
#          [0.0309, 0.9271, 0.5249, 0.0000],
#          [0.3206, 0.2359, 0.5422, 0.0000],
#          [0.3296, 0.1679, 0.5651, 0.0000],
#          [0.3129, 0.0576, 0.5542, 0.0000],
#          [0.4921, 0.1932, 0.7793, 0.0000],
#          [0.5308, 0.0904, 0.7753, 0.0000],
#          [0.4669, 0.0536, 0.7756, 0.0000],
#          [0.2559, 0.2577, 0.4670, 0.0000],
#          [0.2850, 0.1996, 0.5047, 0.0000],
#          [0.2575, 0.0515, 0.5122, 0.0000],
#          [0.3650, 0.2407, 0.6122, 0.0000],
#          [0.3806, 0.0602, 0.6506, 0.0000],
#          [0.3512, 0.5203, 0.5957, 0.0000],
#          [0.3771, 0.4060, 0.6187, 0.0000],
#          [0.3276, 0.4012, 0.5599, 0.0000],
#          [0.4386, 0.1922, 0.7112, 0.0000],
#          [0.4333, 0.0741, 0.7623, 0.0000],
#          [0.2868, 0.3309, 0.4949, 0.0000],
#          [0.4047, 0.2938, 0.6491, 0.0000],
#          [0.2141, 0.3987, 0.4350, 0.0000],
#          [0.5461, 0.2613, 0.8184, 0.0000],
#          [0.5632, 0.1407, 0.8335, 0.0000],
#          [0.6034, 0.1417, 0.8503, 0.0000],
#          [0.4762, 0.3390, 0.7415, 0.0000],
#          [0.3273, 0.3149, 0.5703, 0.0000],
#          [0.4244, 0.3827, 0.6612, 0.0000],
#          [0.5321, 0.4479, 0.7887, 0.0000],
#          [0.2779, 0.4200, 0.5107, 0.0000],
#          [0.3030, 0.4815, 0.5338, 0.0000],
#          [0.4791, 0.5066, 0.7272, 0.0000],
#          [0.4439, 0.4761, 0.6854, 0.0000],
#          [0.2697, 0.5259, 0.4940, 0.0000],
#          [0.4000, 0.5145, 0.6336, 0.0000],
#          [0.5095, 0.5960, 0.7518, 0.0000],
#          [0.5477, 0.5777, 0.8080, 0.0000],
#          [0.2218, 0.6260, 0.4669, 0.0000],
#          [0.3292, 0.5919, 0.5575, 0.0000],
#          [0.4377, 0.5747, 0.6724, 0.0000],
#          [0.4731, 0.6227, 0.7070, 0.0000],
#          [0.3906, 0.6508, 0.6112, 0.0000],
#          [0.3091, 0.7016, 0.5292, 0.0000],
#          [0.4525, 0.7247, 0.6737, 0.0000],
#          [0.5334, 0.7697, 0.7657, 0.0000],
#          [0.3575, 0.7316, 0.5632, 0.0000],
#          [0.3925, 0.8128, 0.5986, 0.0000],
#          [0.2755, 0.9482, 0.5054, 0.0000],
#          [0.2885, 0.7976, 0.5036, 0.0000],
#          [0.2313, 0.8139, 0.4879, 0.0000],
#          [0.4738, 0.8200, 0.6869, 0.0000],
#          [0.4475, 0.9317, 0.6574, 0.0000],
#          [0.5006, 0.8631, 0.7426, 0.0000],
#          [0.3417, 0.8362, 0.5555, 0.0000],
#          [0.4689, 0.8733, 0.6830, 0.0000],
#          [0.4126, 0.9256, 0.6485, 0.0000],
#          [0.3472, 0.9495, 0.5695, 0.0000],
#          [0.4962, 0.9286, 0.6962, 0.0000],
#          [0.5306, 0.9754, 0.7661, 0.0000],
#          [0.8718, 0.9307, 0.8709, 0.0000],
#          [0.8203, 0.6866, 0.7152, 0.0000],
#          [0.7981, 0.1866, 0.7230, 0.0000],
#          [0.7944, 0.0399, 0.7182, 0.0000],
#          [0.8414, 0.1300, 0.7225, 0.0000],
#          [0.8585, 0.0515, 0.7489, 0.0000],
#          [0.9756, 0.1298, 0.7874, 0.0000],
#          [0.9559, 0.0269, 0.7953, 0.0000],
#          [0.9046, 0.0140, 0.7485, 0.0000],
#          [0.8986, 0.1409, 0.7433, 0.0000],
#          [0.8515, 0.2349, 0.7470, 0.0000],
#          [0.8044, 0.4096, 0.6983, 0.0000],
#          [0.8014, 0.2776, 0.7084, 0.0000],
#          [0.9195, 0.2186, 0.7913, 0.0000],
#          [0.9070, 0.3448, 0.7930, 0.0000],
#          [0.9634, 0.2722, 0.8639, 0.0000],
#          [0.8386, 0.3456, 0.7285, 0.0000],
#          [0.8631, 0.3362, 0.7715, 0.0000],
#          [0.8583, 0.4688, 0.7237, 0.0000],
#          [0.8888, 0.4198, 0.7504, 0.0000],
#          [0.7843, 0.5112, 0.6813, 0.0000],
#          [0.8168, 0.5091, 0.6922, 0.0000],
#          [0.9664, 0.5201, 0.7882, 0.0000],
#          [0.8359, 0.5724, 0.6965, 0.0000],
#          [0.7950, 0.5858, 0.6934, 0.0000],
#          [0.8699, 0.6178, 0.7300, 0.0000],
#          [0.9084, 0.5787, 0.7576, 0.0000]])


# }

#     orientation_data = {
#     'Positions': torch.tensor([[0.0217, 0.0427, 0.1651, 0.0000],
#          [0.0101, 0.2308, 0.1256, 0.0000],
#          [0.0423, 0.3136, 0.1666, 0.0000],
#          [0.1599, 0.3202, 0.3689, 0.0000],
#          [0.0787, 0.4736, 0.2016, 0.0000],
#          [0.0240, 0.7489, 0.1160, 0.0000],
#          [0.0835, 0.7704, 0.1963, 0.0000],
#          [0.0569, 0.8164, 0.1551, 0.0000],
#          [0.0363, 0.9721, 0.0113, 0.0000],
#          [0.1052, 0.9540, 0.1480, 0.0000],
#          [0.3815, 0.0213, 0.3299, 0.0000],
#          [0.5969, 0.1145, 0.6141, 0.0000],
#          [0.5682, 0.0447, 0.5560, 0.0000],
#          [0.5904, 0.1978, 0.6004, 0.0000],
#          [0.3153, 0.3101, 0.2094, 0.0000],
#          [0.6391, 0.2885, 0.6903, 0.0000],
#          [0.4162, 0.3812, 0.3053, 0.0000],
#          [0.3963, 0.5102, 0.2674, 0.0000],
#          [0.6892, 0.4231, 0.8022, 0.0000],
#          [0.3840, 0.6752, 0.2269, 0.0000],
#          [0.5689, 0.5178, 0.5201, 0.0000],
#          [0.4692, 0.5419, 0.3519, 0.0000],
#          [0.5777, 0.5899, 0.5391, 0.0000],
#          [0.5086, 0.6468, 0.4023, 0.0000],
#          [0.5273, 0.7445, 0.4308, 0.0000],
#          [0.5904, 0.7651, 0.5508, 0.0000],
#          [0.4631, 0.9906, 0.3010, 0.0000],
#          [0.3102, 0.7984, 0.1791, 0.0000],
#          [0.6643, 0.9370, 0.6866, 0.0000],
#          [0.8766, 0.2181, 0.5177, 0.0000],
#          [0.9487, 0.1907, 0.6126, 0.0000],
#          [0.9719, 0.0988, 0.5646, 0.0000],
#          [0.9045, 0.2999, 0.5467, 0.0000],
#          [0.9373, 0.3903, 0.5926, 0.0000],
#          [0.8557, 0.5500, 0.5025, 0.0000],
#          [0.9869, 0.4540, 0.6628, 0.0000],
#          [0.9658, 0.5295, 0.6065, 0.0000],
#          [0.9150, 0.5432, 0.5406, 0.0000],
#          [0.9929, 0.6059, 0.6469, 0.0000],
#          [0.9769, 0.0262, 0.5539, 0.0000],
#          [0.0605, 0.0295, 0.4032, 0.0000],
#          [0.1069, 0.0292, 0.4982, 0.0000],
#          [0.0242, 0.2961, 0.3127, 0.0000],
#          [0.0603, 0.2981, 0.3633, 0.0000],
#          [0.0596, 0.4606, 0.3281, 0.0000],
#          [0.0290, 0.6204, 0.2730, 0.0000],
#          [0.1142, 0.5574, 0.4595, 0.0000],
#          [0.0824, 0.5851, 0.3943, 0.0000],
#          [0.0741, 0.7763, 0.3626, 0.0000],
#          [0.1237, 0.9464, 0.4213, 0.0000],
#          [0.1640, 0.9533, 0.4857, 0.0000],
#          [0.3276, 0.0000, 0.4404, 0.0000],
#          [0.5018, 0.1028, 0.6544, 0.0000],
#          [0.5720, 0.1506, 0.7284, 0.0000],
#          [0.5402, 0.1648, 0.6938, 0.0000],
#          [0.5669, 0.2303, 0.7202, 0.0000],
#          [0.6478, 0.4269, 0.8760, 0.0000],
#          [0.5726, 0.3456, 0.7561, 0.0000],
#          [0.4558, 0.4213, 0.5466, 0.0000],
#          [0.6165, 0.3928, 0.8091, 0.0000],
#          [0.3952, 0.5041, 0.4645, 0.0000],
#          [0.4927, 0.5711, 0.5722, 0.0000],
#          [0.6377, 0.6295, 0.8213, 0.0000],
#          [0.5915, 0.5940, 0.7549, 0.0000],
#          [0.5759, 0.6646, 0.6915, 0.0000],
#          [0.2547, 0.8073, 0.3407, 0.0000],
#          [0.2165, 0.7763, 0.3340, 0.0000],
#          [0.3752, 0.8271, 0.3674, 0.0000],
#          [0.4709, 0.9695, 0.4813, 0.0000],
#          [0.3954, 0.9779, 0.3958, 0.0000],
#          [0.8638, 0.2052, 0.6692, 0.0000],
#          [0.8199, 0.3812, 0.6255, 0.0000],
#          [0.8296, 0.4695, 0.6208, 0.0000],
#          [0.9891, 0.6453, 0.7681, 0.0000],
#          [0.8780, 0.5056, 0.6477, 0.0000],
#          [0.8349, 0.5914, 0.6048, 0.0000],
#          [0.9092, 0.7047, 0.6879, 0.0000],
#          [0.7395, 0.7923, 0.5294, 0.0000],
#          [0.7497, 0.9276, 0.5430, 0.0000],
#          [0.7803, 0.9215, 0.6216, 0.0000],
#          [0.0416, 0.0122, 0.5624, 0.0000],
#          [0.1067, 0.1907, 0.5887, 0.0000],
#          [0.0565, 0.2128, 0.5603, 0.0000],
#          [0.0505, 0.2938, 0.5431, 0.0000],
#          [0.0550, 0.4975, 0.5553, 0.0000],
#          [0.0817, 0.5805, 0.5729, 0.0000],
#          [0.0285, 0.6539, 0.5305, 0.0000],
#          [0.0974, 0.7428, 0.5833, 0.0000],
#          [0.0629, 0.8220, 0.5581, 0.0000],
#          [0.1589, 0.9657, 0.5854, 0.0000],
#          [0.3129, 0.0576, 0.5542, 0.0000],
#          [0.4669, 0.0536, 0.7756, 0.0000],
#          [0.2575, 0.0515, 0.5122, 0.0000],
#          [0.3512, 0.5203, 0.5957, 0.0000],
#          [0.4386, 0.1922, 0.7112, 0.0000],
#          [0.4047, 0.2938, 0.6491, 0.0000],
#          [0.5632, 0.1407, 0.8335, 0.0000],
#          [0.3273, 0.3149, 0.5703, 0.0000],
#          [0.2779, 0.4200, 0.5107, 0.0000],
#          [0.6137, 0.5485, 0.8417, 0.0000],
#          [0.4000, 0.5145, 0.6336, 0.0000],
#          [0.2218, 0.6260, 0.4669, 0.0000],
#          [0.4731, 0.6227, 0.7070, 0.0000],
#          [0.4525, 0.7247, 0.6737, 0.0000],
#          [0.3925, 0.8128, 0.5986, 0.0000],
#          [0.2313, 0.8139, 0.4879, 0.0000],
#          [0.5006, 0.8631, 0.7426, 0.0000],
#          [0.6330, 0.9601, 0.8088, 0.0000],
#          [0.4962, 0.9286, 0.6962, 0.0000],
#          [0.8203, 0.6866, 0.7152, 0.0000],
#          [0.7525, 0.2755, 0.7160, 0.0000],
#          [0.8414, 0.1300, 0.7225, 0.0000],
#          [0.9559, 0.0269, 0.7953, 0.0000],
#          [0.8515, 0.2349, 0.7470, 0.0000],
#          [0.9195, 0.2186, 0.7913, 0.0000],
#          [0.8386, 0.3456, 0.7285, 0.0000],
#          [0.8888, 0.4198, 0.7504, 0.0000],
#          [0.9664, 0.5201, 0.7882, 0.0000],
#          [0.8699, 0.6178, 0.7300, 0.0000]]),
#  'Values': torch.tensor([[-0.4406,  0.0410,  0.8968,  0.0000],
#          [-0.4470,  0.0922,  0.8898,  0.0000],
#          [-0.4658,  0.0777,  0.8815,  0.0000],
#          [-0.2454,  0.2422,  0.9387,  0.0000],
#          [-0.4345,  0.0786,  0.8972,  0.0000],
#          [-0.4185,  0.0155,  0.9081,  0.0000],
#          [-0.2924,  0.1258,  0.9480,  0.0000],
#          [-0.3167,  0.1222,  0.9406,  0.0000],
#          [-0.1892,  0.6801,  0.7082,  0.0000],
#          [-0.3618,  0.3807,  0.8510,  0.0000],
#          [-0.3576,  0.1149,  0.9268,  0.0000],
#          [-0.4498, -0.0028,  0.8931,  0.0000],
#          [-0.4390, -0.0387,  0.8976,  0.0000],
#          [-0.4480,  0.0041,  0.8940,  0.0000],
#          [-0.3332,  0.0768,  0.9397,  0.0000],
#          [-0.4898, -0.0607,  0.8697,  0.0000],
#          [-0.3192,  0.1385,  0.9375,  0.0000],
#          [-0.3032,  0.1112,  0.9464,  0.0000],
#          [-0.5120, -0.0025,  0.8590,  0.0000],
#          [-0.3001,  0.0386,  0.9531,  0.0000],
#          [-0.5203,  0.0736,  0.8508,  0.0000],
#          [-0.4298,  0.0764,  0.8997,  0.0000],
#          [-0.5177,  0.0357,  0.8548,  0.0000],
#          [-0.4654,  0.0591,  0.8831,  0.0000],
#          [-0.4819,  0.0467,  0.8750,  0.0000],
#          [-0.5091,  0.1267,  0.8513,  0.0000],
#          [-0.3988,  0.1092,  0.9105,  0.0000],
#          [-0.0618, -0.0505,  0.9968,  0.0000],
#          [-0.5847,  0.0864,  0.8066,  0.0000],
#          [-0.1999, -0.0543,  0.9783,  0.0000],
#          [-0.4835, -0.2388,  0.8421,  0.0000],
#          [-0.5063, -0.3744,  0.7769,  0.0000],
#          [-0.3571,  0.0226,  0.9338,  0.0000],
#          [-0.4242,  0.0956,  0.9005,  0.0000],
#          [-0.1612,  0.0152,  0.9868,  0.0000],
#          [-0.4378,  0.1025,  0.8932,  0.0000],
#          [-0.4658,  0.0524,  0.8834,  0.0000],
#          [-0.2245,  0.0545,  0.9730,  0.0000],
#          [-0.3967,  0.0599,  0.9160,  0.0000],
#          [-0.4910, -0.2054,  0.8466,  0.0000],
#          [-0.5020,  0.0434,  0.8638,  0.0000],
#          [-0.4754,  0.0913,  0.8750,  0.0000],
#          [-0.4223,  0.1177,  0.8988,  0.0000],
#          [-0.4163,  0.1138,  0.9021,  0.0000],
#          [-0.4763,  0.0674,  0.8767,  0.0000],
#          [-0.4880, -0.0065,  0.8728,  0.0000],
#          [-0.5806, -0.0284,  0.8137,  0.0000],
#          [-0.6448, -0.0471,  0.7629,  0.0000],
#          [-0.5270,  0.0648,  0.8474,  0.0000],
#          [-0.4373,  0.1412,  0.8882,  0.0000],
#          [-0.4465,  0.1375,  0.8842,  0.0000],
#          [-0.3039,  0.0667,  0.9504,  0.0000],
#          [-0.3546,  0.0475,  0.9338,  0.0000],
#          [-0.3684,  0.0161,  0.9295,  0.0000],
#          [-0.3322,  0.0217,  0.9430,  0.0000],
#          [-0.3667, -0.0664,  0.9280,  0.0000],
#          [-0.4689,  0.0108,  0.8832,  0.0000],
#          [-0.4509, -0.0317,  0.8920,  0.0000],
#          [-0.3733,  0.1213,  0.9197,  0.0000],
#          [-0.3959, -0.0763,  0.9151,  0.0000],
#          [-0.3090,  0.1585,  0.9378,  0.0000],
#          [-0.4135,  0.1894,  0.8906,  0.0000],
#          [-0.4744,  0.1442,  0.8684,  0.0000],
#          [-0.3970,  0.2533,  0.8822,  0.0000],
#          [-0.5080,  0.2240,  0.8317,  0.0000],
#          [-0.0203, -0.2459,  0.9691,  0.0000],
#          [ 0.0228, -0.2057,  0.9784,  0.0000],
#          [-0.2428,  0.0597,  0.9682,  0.0000],
#          [-0.3982, -0.0275,  0.9169,  0.0000],
#          [-0.3217, -0.0237,  0.9465,  0.0000],
#          [-0.3650,  0.0233,  0.9307,  0.0000],
#          [-0.2800,  0.0539,  0.9585,  0.0000],
#          [-0.2087,  0.1449,  0.9672,  0.0000],
#          [-0.3394,  0.0479,  0.9394,  0.0000],
#          [-0.2896,  0.1400,  0.9468,  0.0000],
#          [-0.2213, -0.0030,  0.9752,  0.0000],
#          [-0.2863, -0.0602,  0.9562,  0.0000],
#          [-0.3844, -0.0192,  0.9230,  0.0000],
#          [-0.5866,  0.0574,  0.8079,  0.0000],
#          [-0.4417, -0.0399,  0.8963,  0.0000],
#          [-0.2364,  0.0906,  0.9674,  0.0000],
#          [-0.0482,  0.0599,  0.9970,  0.0000],
#          [-0.2488,  0.0301,  0.9681,  0.0000],
#          [-0.1798,  0.0925,  0.9793,  0.0000],
#          [-0.2397,  0.0121,  0.9708,  0.0000],
#          [-0.1886, -0.0130,  0.9820,  0.0000],
#          [-0.2950,  0.0039,  0.9555,  0.0000],
#          [-0.1277,  0.0430,  0.9909,  0.0000],
#          [-0.2593,  0.0217,  0.9656,  0.0000],
#          [-0.0536,  0.0289,  0.9981,  0.0000],
#          [-0.3353,  0.0965,  0.9372,  0.0000],
#          [-0.1892, -0.1052,  0.9763,  0.0000],
#          [-0.1670,  0.1520,  0.9742,  0.0000],
#          [-0.3117,  0.0394,  0.9494,  0.0000],
#          [-0.3506,  0.1689,  0.9212,  0.0000],
#          [-0.3370,  0.0859,  0.9376,  0.0000],
#          [-0.2380, -0.2033,  0.9498,  0.0000],
#          [-0.4188, -0.0153,  0.9079,  0.0000],
#          [-0.3199, -0.0572,  0.9457,  0.0000],
#          [-0.1748,  0.0513,  0.9833,  0.0000],
#          [-0.2896,  0.0382,  0.9564,  0.0000],
#          [-0.1941, -0.0303,  0.9805,  0.0000],
#          [-0.3342,  0.0701,  0.9399,  0.0000],
#          [-0.3322,  0.0728,  0.9404,  0.0000],
#          [-0.3094, -0.0331,  0.9504,  0.0000],
#          [-0.0903, -0.0495,  0.9947,  0.0000],
#          [-0.4587,  0.0781,  0.8851,  0.0000],
#          [-0.1615,  0.0376,  0.9862,  0.0000],
#          [-0.4090,  0.0625,  0.9104,  0.0000],
#          [-0.2219, -0.1400,  0.9650,  0.0000],
#          [ 0.0105,  0.0722,  0.9973,  0.0000],
#          [-0.0945,  0.0033,  0.9955,  0.0000],
#          [-0.2189,  0.0658,  0.9735,  0.0000],
#          [-0.2146, -0.0784,  0.9736,  0.0000],
#          [-0.2378, -0.2154,  0.9471,  0.0000],
#          [-0.3108,  0.0815,  0.9470,  0.0000],
#          [-0.1860,  0.1700,  0.9677,  0.0000],
#          [-0.2414,  0.0476,  0.9693,  0.0000],
#          [-0.2618, -0.1514,  0.9532,  0.0000]])
# }

    gp.interface_data(interface_data)
    gp.orientation_data(orientation_data)
    gp.interpolation_options()
    gp.interpolation_options_set(Transformation_matrix=Transformation_matrix)
    # custom_data = torch.tensor([[40,20,30,0]], dtype=torch.float32)

    # gp.activate_custom_grid(custom_grid_data=custom_data)
    #gp.active_grid()
    #gp.deactivate_grid("Regular")
    sol = gp.Solution()
    # print(sol)
 
    #gp.active_grid()
    #gp.activate_grid("Regular")
    gp.active_grid()
    sol = gp.Solution()

    #########################################################################
    ###### Uncomment the below code lines for matplotlib visualization ######
    #########################################################################

    ##### FOR 2D matplotlib #####
    import time
    for t in [-0.5, 0, 0.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3, 3.5, 4,4.5]:
        gp.plot_data_section(section={2:0.5, 4:t}, plot_scalar_field = True, plot_input_data=True)
        time.sleep(1)


    ##### FOR 3D matplotlib #####
    import time
    for t in [-0.5, 0, 0.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3, 3.5, 4,4.5]:
        gp.plot_data_section(section={4:t}, plot_scalar_field = True, plot_input_data=True)
        time.sleep(1)


    #############################################################################################
    ########## Uncomment below for  Interactive Visualization Pyvista below  ####################
    #############################################################################################
    
    ###############################################################
    ########### show/unshow input data using "plot_input_data" argument
    ########### show/unshow surface or interfaces using "only_surface_mode" argument
    ###############################################################

    print("\nStarting Interactive Visualization...")
    gp.plot_interactive_section(plot_input_data = True, only_surface_mode = False)

    
if __name__ == "__main__":
    main()