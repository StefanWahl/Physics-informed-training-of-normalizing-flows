import torch
from typing import Callable
import matplotlib.pyplot as plt

#Create coordinate grid
def get_grid_n_dim(res_list:list,lim_list:list):

    '''
    parameters:
        res_list: List of integers containing the number of grid points along the different dimensions
        lim_list: Lists of lists contaiing the limits of the gird along the different dimensions

    returns:
        grid_points:            Tensor of shape (N,d) containing the grid points
        spacings_tensor:        List containing the distance between grid points for each dimension
        coordinate_grids_list:  List containing the coordinate grids for each dimension
    '''

    d = len(res_list)

    #get ranges for the different dimensions
    range_list = [torch.linspace(lim_list[i][0],lim_list[i][1],res_list[i]) for i in range(d)]

    #Get the spacings between two points
    spacings_tensor = torch.zeros(d)

    for i in range(d):
        spacings_tensor[i] = range_list[i][1] - range_list[i][0]

    #Get grids for the different dimensions
    coordinate_grids = torch.meshgrid(range_list,indexing="xy")

    #Combine the grids
    coordinate_grids_list = []

    for i in range(d):
        coordinate_grids_list.append(coordinate_grids[i].reshape(-1,1))

    grid_points = torch.cat(coordinate_grids_list,-1)

    return grid_points,spacings_tensor,coordinate_grids

#Evaluate pdf on a grid
def eval_pdf_on_grid_2D(pdf:Callable,x_lims:list = [-10.0,10.0],y_lims:list = [-10.0,10.0],x_res:int = 200,y_res:int = 200,device = "cpu",args_pdf = (),kwargs_pdf = {}):
    """
    parameters:
        pdf:        Probability density function
        x_lims:     Limits of the evaluated region in x directions
        y_lims:     Limits of the evaluated region in y directions
        x_res:      Number of grid points in x direction
        y_res:      Number of grid points in y direction

    returns:
        pdf_grid:   Grid of pdf values
        x_grid:     Grid of x coordinates
        y_grid:     Grid of y coordinates
    """

    grid_points,spacings_tensor,coordinate_grids = get_grid_n_dim(res_list = [x_res,y_res],lim_list = [x_lims,y_lims])

    #Evaluate the pdf
    pdf_grid = pdf(grid_points.to(device),*args_pdf,**kwargs_pdf).reshape(y_res,x_res)

    x_grid = coordinate_grids[0]
    y_grid = coordinate_grids[1]

    return pdf_grid,x_grid,y_grid

#Visualize the pdf 
def plot_pdf_2D(pdf_grid:torch.tensor,x_grid:torch.tensor,y_grid:torch.tensor,ax:plt.axes,fs:int = 20,title:str = "",range_vals = [None,None],cmap:str = "viridis",return_im:bool = False,turn_off_axes:bool = False):
    """
    parameters:
        pdf_grid:       Grid of pdf values
        x_grid:         Grid of x coordinates
        y_grid:         Grid of y coordinates
        ax:             Axes for plotting
        fs:             Fontsize
        title:          Title of the plot
        range_vals:     Range of the colorbar
        cmap:           Colormap
        return_im:      Return the image object
        turn_off_axes:  Turn off the axes
    """

    im = ax.pcolormesh(x_grid,y_grid,pdf_grid,vmin = range_vals[0],vmax = range_vals[1],cmap = cmap)
    ax.set_xlabel("x",fontsize = fs)
    ax.set_ylabel("y",fontsize = fs)
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    ax.set_title(title,fontsize = fs)

    if turn_off_axes:
        ax.axis("off")

    plt.tight_layout()

    if return_im:
        return im
