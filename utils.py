# Import the necessary libraries
import cdsapi
import copy
import xarray as xr
import calendar
import cartopy.crs as ccrs
import cartopy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import numpy as np
import warnings
import copy

from ochanticipy import (
    create_custom_country_config,
    CodAB,
    GeoBoundingBox,
    IriForecastDominant,
    IriForecastProb,
    ChirpsMonthly,
)


# Months and seasons variables and functions

months = [m[0] for m in list(calendar.month_name)[1:]]
months = months + months[:2]
seasons = [
    f"{months[i]}{months[i+1]}{months[i+2]}" for i in range(len(months) - 2)
    ]


def extend_list(lst_initial, lst, length):
    """
    Extend a list up to a certain length.

    This function takes a list (lst_initial) and extends it by repeating the
    values in lst until the list reaches a certain length (length). If lst is
    longer than the amount of values needed to reach length, only the necessary
    amount of values are used. This is used to create a list of seasons,
    starting from the month at which the data starts.

    Inputs
    --------
    lst_initial (list): The initial list to be extended.
    lst (list): The values to be used to extend the initial list.
    length (int): The desired length of the extended list.

    Returns
    --------
    extended_list (list): The extended list.
    """
    return (
        lst_initial + (lst * ((length + len(lst) - 1) // len(lst)))
        )[:length]


def from_monthly_to_seasonal(ds, iri_data=False, retain_year=False):
    """
    Resample monthly precipitation data to seasonal.

    This function resamples monthly precipitation data (ds.tprate) to seasonal
    data by taking the rolling average over three consecutive months. The
    resulting data array is converted to a dataset and returned.

    Inputs
    --------
    ds (xarray Dataset): A dataset with monthly precipitation data to be
        resampled.
    iri_data (bool): Deprecated argument. Does not affect the function.
    retain_year (bool): If True, the year information is retained by appending
        it to the season names. If False, only the season names are used.

    Returns
    --------
    ds_seas (xarray Dataset): The resampled seasonal precipitation data.
    """
    months_init = months[ds.time.dt.month.values[0] - 1 :]
    seasons_init = [
        f"{months_init[i]}{months_init[i+1]}{months_init[i+2]}"
        for i in range(len(months_init) - 2)
    ]

    ds_seas = ds.tprate.rolling(time=3, center=True).mean()

    if retain_year:
        ds_seas.coords["time"] = [
            f"{s}-{y}"
            for (s, y) in zip(
                extend_list(seasons_init, seasons, len(ds_seas["time"])),
                ds_seas.coords["time"].dt.year.values,
            )
        ]
    else:
        ds_seas.coords["time"] = extend_list(
            seasons_init, seasons, len(ds_seas["time"])
        )

    return ds_seas.to_dataset()


# Coarse resolution

def coarser_coord_by_amount(ds, coord_name, factor, method="linear"):
    """
    Coarsens a given coordinate in a dataset by a given factor.

    Given a dataset and a coordinate, this function reduces the number of
    points in the coordinate by a given factor by interpolating the values in
    between the original values.

    Inputs
    --------
    ds (xarray.Dataset): the dataset to be coarsened.
    coord_name (str): the name of the coordinate to be coarsened.
    factor (int): the factor by which the coordinate should be coarsened.
    method (str, optional): the interpolation method to be used. Defaults to
        'linear'.

    Returns
    --------
    ds_coarser (xarray.Dataset): the coarsened dataset with interpolated
        values.
    """
    if not coord_name in ds.coords:
        raise ValueError(
            f"The Dataset does not contain the coordinate {coord_name}."
            )
    old_coord = ds[coord_name].values.tolist()
    ind_list = [
        int(factor / 2 + n * factor)
        for n in range(len(old_coord))
        if factor / 2 + n * factor < len(old_coord)
    ]
    if factor % 2:
        new_coord = [old_coord[i] for i in ind_list]
    else:
        new_coord = [(old_coord[i] + old_coord[i - 1]) / 2 for i in ind_list]
    ds_coarser = ds.interp(
        coords={coord_name: np.array(new_coord)}, 
        method=method
        )
    return ds_coarser.interpolate_na


def coarser_coords_adapt_do_ds(ds, ds_ref, coord_names, method="linear"):
    """
    Adapt the coordinates of a dataset to a reference dataset by coarsening.

    This function takes in a dataset (ds), a reference dataset (ds_ref), a list
    of coordinate names (coord_names), and an optional method (default is
    'linear'). It returns a copy of the input dataset with its coordinates
    adapted to the resolution of the reference dataset.

    Inputs
    --------
    ds (xarray.Dataset): the dataset to be adapted
    ds_ref (xarray.Dataset): the reference dataset
    coord_names (list of str): list of coordinate names to be adapted
    method (str): the interpolation method used to coarsen the coordinates.
        Default to 'linear'.

    Returns
    --------
    ds (xarray.Dataset): the adapted dataset
    """
    for coord_name in coord_names:
        if not coord_name in ds_ref.coords:
            raise ValueError(
                "The reference Dataset does not contain the coordinate "
                f"{coord_name}."
            )
        if abs(
            np.diff(ds_ref[coord_name])[0]
            ) < abs(np.diff(ds[coord_name])[0]):
            raise ValueError(
                "The resolution of the reference Dataset is lower than the "
                "one of your Dataset!"
            )
        if (
            not abs(
            np.diff(ds_ref[coord_name])[0] / np.diff(ds[coord_name])[0]
            ) % 1
            > 0.001
        ):
            raise ValueError(
                "The resolution of the reference Dataset is not a multiple of "
                "the one of your Dataset!"
            )
        factor = abs(
            int(np.diff(ds_ref[coord_name])[0] / np.diff(ds[coord_name])[0])
            )
        if factor == 1:
            warnings.warn("The two datasets have the same resolution")
            ds = ds.copy()
        else:
            ds = coarser_coord_by_amount(
                ds=ds, coord_name=coord_name, factor=factor, method=method
            )
    return ds


# Shift coordinates
def shift_coord_by_amount(ds, coord_name, shift_amount, method="linear"):
    """
    Shift the values of a coordinate in a dataset by a specified amount.

    This function takes in a dataset (ds), a coordinate name (coord_name), a
    shift amount (shift_amount), and an optional method (default is 'linear').
    It returns a copy of the input dataset with the values of the specified
    coordinate shifted by the specified amount.

    Inputs
    --------
    ds (xarray.Dataset): the dataset to be shifted coord_name (str): the name
        of the coordinate to be shifted shift_amount (float): the amount by 
        which to shift the coordinate values method (str): the interpolation 
        method used to shift the coordinates (default is 'linear').

    Returns
    --------
    ds_shifted (xarray.Dataset): the shifted dataset.
    """
    if not coord_name in ds.coords:
        raise ValueError(
            f"The Dataset does not contain the coordinate {coord_name}."
            )
    new_coord = ds[coord_name] + shift_amount
    ds_shifted = ds.interp(coords={coord_name: new_coord}, method=method)
    ds_shifted[coord_name] = new_coord
    return ds_shifted


def shift_coords_adapt_do_ds(ds, ds_ref, coord_names, method="linear"):
    """
    Shift the coordinates of `ds` to match those of `ds_ref`.

    Return a new dataset with updated coordinates.

    Inputs
    ------
    ds (xarray.Dataset): The dataset to be shifted.
    ds_ref (xarray.Dataset): The reference dataset to which `ds` coordinates
        will be shifted.
    coord_names (list): A list of the names of the coordinates to be shifted.
    method (str): The interpolation method to use during shifting. Defaults to
        'linear'.

    Returns
    -------
    new_ds (xarray.Dataset): The updated dataset with shifted coordinates.
    """
    for coord_name in coord_names:
        if not coord_name in ds_ref.coords:
            raise ValueError(
                f"The reference Dataset does not contain the "
                f"coordinate {coord_name}."
            )
        ds_res = abs(np.diff(ds[coord_name])[0])
        ds_ref_res = abs(np.diff(ds_ref[coord_name])[0])
        if ds_res != ds_ref_res:
            raise ValueError(
                "The resolutions of the two Datasets are different!"
                )
        shift_amount = ds_ref[coord_name][0] - ds[coord_name][0]
        if shift_amount == 0:
            warnings.warn("The two datasets have the same coordinates.")
            new_ds = ds.copy()
        else:
            new_ds = shift_coord_by_amount(
                ds=ds, 
                coord_name=coord_name, 
                shift_amount=shift_amount, 
                method=method
            )
    return new_ds


def coords_adapt_do_ds(
        ds, 
        ds_ref, 
        coord_names, 
        two_step_comp=False, 
        method="linear"
        ):
    """
    Adjusts the coordinates of the input dataset `ds` to match those of the
    reference dataset `ds_ref` and returns a new dataset with updated
    coordinates.

    Inputs
    ------
    ds (xarray.Dataset): The dataset to be shifted.
    ds_ref (xarray.Dataset): The reference dataset to which `ds` coordinates
        will be shifted.
    coord_names (list): A list of names of the coordinates to be adjusted.
    two_step_comp (bool): If True, the function first coarsens the coordinates
        and then adjusts them. Default is False.
    method (str): The interpolation method to use during coordinate adjustment.
        Default is 'linear'.

    Returns
    -------
    new_ds (xarray.Dataset): A new dataset with adjusted coordinates.

    Raises
    ------
    ValueError: If a coordinate from `coord_names` is not present in
        `ds_ref`.
    """
    if two_step_comp:
        ds_coarse = coarser_coords_adapt_do_ds(
            ds=ds, ds_ref=ds_ref, coord_names=coord_names, method=method
        )
        new_ds = shift_coords_adapt_do_ds(
            ds=ds_coarse, ds_ref=ds_ref, coord_names=coord_names, method=method
        )
    else:
        new_ds = ds
        for coord_name in coord_names:
            if coord_name not in ds_ref.coords:
                raise ValueError(
                    f"The reference dataset does not contain the "
                    f"coordinate {coord_name}."
                )
            new_coord = ds_ref[coord_name].values
            new_ds = new_ds.interp(
                coords={coord_name: new_coord}, 
                method=method
                )
    return new_ds


# Areas

def create_area(country_config_file):
    """
    Create a GeoBoundingBox object and a custom country configuration object.

    This function creates a custom country configuration object from the input
    file path and uses it to create a CodAB object. It then downloads the
    administrative boundary data from the CodAB object and uses it to create a
    GeoBoundingBox object.

    Inputs
    -------
    country_config_file (str): The path to the custom country configuration
        file.

    Returns
    -------
    country_config (object): A custom country configuration object.
    geobb (object): A GeoBoundingBox object created from the administrative
        boundary data.
    """
    country_config = create_custom_country_config(country_config_file)
    codab = CodAB(country_config=country_config)
    codab.download(clobber=True)
    gdf_adm0 = codab.load(admin_level=0)
    geobb = GeoBoundingBox.from_shape(gdf_adm0)
    return country_config, geobb


def increase_box(geobb, size=1):
    """
    Increase the size of a GeoBoundingBox object.

    This function creates a copy of the input GeoBoundingBox object and 
    increases its size by the specified amount (in degrees) in all four 
    directions.

    Inputs
    -------
    geobb (object): A GeoBoundingBox object.
    size (float): The amount (in degrees) by which to increase the size of the
        GeoBoundingBox object. Defaults to 1 degree.

    Returns
    -------
    geobb_enlarged (object): A copy of the input GeoBoundingBox object with 
        increased size.
    geobb (object): The original GeoBoundingBox object.
    """
    geobb_enlarged = copy.deepcopy(geobb)
    geobb_enlarged.lat_max = geobb.lat_max + size
    geobb_enlarged.lat_min = geobb.lat_min - size
    geobb_enlarged.lon_max = geobb.lon_max + size
    geobb_enlarged.lon_min = geobb.lon_min - size
    return geobb_enlarged, geobb


def restrict_to_geobb(ds, geobb):
    """
    Restrict a dataset to a given geographic bounding box.

    This function selects the portion of the dataset that falls within the
    bounds of a given geographic bounding box, specified by its maximum and
    minimum latitude and longitude values.

    Inputs
    --------
    ds (xarray.Dataset): The dataset to be restricted.
    geobb (namedtuple): A namedtuple containing the following fields:
        lat_max (float): The maximum latitude of the bounding box.
        lat_min (float): The minimum latitude of the bounding box.
        lon_max (float): The maximum longitude of the bounding box.
        lon_min (float): The minimum longitude of the bounding box.

    Returns
    --------
    ds_restricted (xarray.Dataset): The restricted dataset, containing only
        the data that falls within the specified geographic bounding box.
    """
    ds_restricted = ds.sel(
        latitude=slice(geobb.lat_max, geobb.lat_min),
        longitude=slice(geobb.lon_min, geobb.lon_max),
    )
    return ds_restricted


# Figures

def create_fig():
    """
    Create a figure object with a specified layout and projection.

    This function creates a figure object with a specified layout using
    the GridSpec functionality. It then creates four subplots within this
    layout, with the fourth subplot having a specified projection using
    the PlateCarree coordinate reference system. Finally, it adds borders
    to this subplot.

    Returns
    --------
    fig (matplotlib.figure.Figure): The figure object created.
    ax1 (matplotlib.axes.Axes): The first subplot object.
    ax2 (matplotlib.axes.Axes): The second subplot object.
    ax3 (matplotlib.axes.Axes): The third subplot object.
    ax4 (cartopy.mpl.geoaxes.GeoAxes): The fourth subplot object, with
        PlateCarree projection and borders added.
    """
    fig = plt.figure(constrained_layout=True)

    gs = GridSpec(10, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1:, :], projection=ccrs.PlateCarree())

    ax4.add_feature(cartopy.feature.BORDERS, edgecolor="black")

    return fig, ax1, ax2, ax3, ax4


def create_cmaps():
    """
    Create variables for creating heatmaps.
    """
    # specify the color boundaries
    bounds = [0.35, 0.55, 0.75, 0.95]

    # colormaps lower
    colors_list = ["#ccece6", "#66c2a4", "#2ca25f"]
    discrete_cmap2 = colors.ListedColormap(colors_list)
    norm2 = colors.BoundaryNorm(bounds, discrete_cmap2.N)

    # colormaps middle
    colors_list = ["#edf8fb", "#edf8fb", "#edf8fb"]
    discrete_cmap1 = colors.ListedColormap(colors_list)
    norm1 = colors.BoundaryNorm(bounds, discrete_cmap1.N)

    # colormaps upper
    colors_list = ["#9ebcda", "#8c96c6", "#8856a7"]
    discrete_cmap0 = colors.ListedColormap(colors_list)
    norm0 = colors.BoundaryNorm(bounds, discrete_cmap0.N)

    return (
        norm0, 
        norm1, 
        norm2, 
        discrete_cmap0, 
        discrete_cmap1, 
        discrete_cmap2, 
        bounds
    )


# Dominant tercile

def xarray_argmax_value(dataArray, coord_name):
    """
    Values corresponding to the index or indices of the maximum of the 
        DataArray over one or more dimensions (coordinates).

    If there are multiple maxima, the indices of the first one found will be 
        returned.

    Inputs:
    --------
    dataArray (xarray.DataArray): input DataArray.
    coord_name (str): name of the coordinate (and dimension) along which the 
        maximum will be calculated.

    Returns
    --------
    dataArray (xarray.DataArray): output DataArray.
    """
    if type(dataArray) == xr.core.dataarray.Dataset:
        dataArray = dataArray.to_array()
    dataArray = dataArray.fillna(-999)
    argmax = dataArray.argmax(coord_name)
    try:
        argmax = argmax.isel(variable=0).drop("variable")
    except:
        pass
    coord_values_list = dataArray.coords[coord_name].values.tolist()
    coord_dict = {i: coord_values_list[i] \
                  for i in range(len(coord_values_list))}
    u, inv = np.unique(argmax, return_inverse=True)
    dataArray = xr.DataArray(
        np.array([coord_dict[j] for j in u])[inv].reshape(argmax.shape),
        dims=argmax.dims,
        coords=argmax.coords,
    )
    return dataArray.where(dataArray != -999)


def calculate_dominant_tercile(ds_terciles):
    """
    Extract dominant tercile from tercile dataset.
    
    Inputs:
    --------
    ds_terciles (xarray.Dataset): Dataset containing information on tercile
        probabilities and dominant tercile (lower, middle or upper) for certain
        values of latitude, longitude and season.

    Returns
    --------
    ds_terciles_dominant (xarray.Dataset): Dataset containing information on 
        the tercile probabilities of the dominant tercile (the coordinate 
        'tercile' was removed by the dataset).
    """
    ds_terciles_dominant = ds_terciles.max(
        "tercile"
        ).rename({"tprate": "terc_prob"})
    ds_terciles_dominant["dominant_terc"] = xarray_argmax_value(
        ds_terciles, 
        "tercile"
        )
    return ds_terciles_dominant


# Chirps

def chirps_terciles_create(
    country_config, 
    geobb, 
    adapt_coordinates=False, 
    ds_ref=None, 
    method="linear"
):
    """
    Download and process CHIRPS data to create tercile probabilities.

    Inputs
    --------
    country_config (dict): dictionary with information about the country,
        including data directories and variable names.
    geobb (tuple): tuple with the bounding box of the region of interest, in
        the format (lon_min, lat_min, lon_max, lat_max).
    adapt_coordinates (bool): whether to adapt the coordinates of the data to
        match the reference dataset. Default is False.
    ds_ref (xarray.Dataset): reference dataset to match the coordinates to.
        Required if adapt_coordinates is True.
    method (str): interpolation method to use when adapting the coordinates.
        Default is 'linear'.

    Returns
    --------
    tercile_probs (xarray.Dataset): tercile probabilities for the forecast 
        data.
    """
    if adapt_coordinates:
        geobb, geobb_original = increase_box(geobb)
    print("Download and process data...")
    chirps_monthly = chirps_download_and_process(country_config, geobb)
    print("Load data...")
    ds_chirps = chirps_load(chirps_monthly)
    ds_chirps["tprate"] = ds_chirps["tprate"].load()
    if adapt_coordinates:
        print("Adapt coordinates...")
        if ds_ref is None:
            raise ValueError(
                "If you want to adapt coordinates you need to provide "
                "the parameter `ds_ref`."
            )
        ds_chirps = coords_adapt_do_ds(
            ds=ds_chirps,
            ds_ref=ds_ref,
            coord_names=["latitude", "longitude"],
            method=method,
        )
        ds_chirps = restrict_to_geobb(ds_chirps, geobb_original)
    print("Create climatology...")
    ds_chirps_seas_reference = chirps_create_reference(ds_chirps)
    print("Calculate quantiles...")
    chirps_quantiles_seas = chirps_quantile_boundaries(
        ds_chirps_seas_reference, seasons
    )
    print("Calculate tercile probabilities...")
    return chirps_tercile_probs(ds_chirps, chirps_quantiles_seas)


def chirps_terciles_plot(ds_chirps_terciles, time):
    """
    Create a plot of the precipitation terciles for the given time.

    Inputs
    -------
    ds_chirps_terciles (xarray Dataset): The CHIRPS dataset.
    time (str): The time for which the plot is to be created.

    Returns
    --------
    fig (matplotlib Figure): The matplotlib figure object of the created plot.
    """
    max_index = ds_chirps_terciles.argmax("tercile")

    fig, ax1, ax2, ax3, ax4 = create_fig()
    (
        norm0,
        norm1,
        norm2,
        discrete_cmap0,
        discrete_cmap1,
        discrete_cmap2,
        bounds,
    ) = create_cmaps()

    c0 = (
        ds_chirps_terciles.isel(tercile=0, drop=True)
        .where(max_index == 0)
        .sel(time=time)
        .tprate.plot.pcolormesh(
            ax=ax4, add_colorbar=False, cmap=discrete_cmap0, norm=norm0
        )
    )
    c1 = (
        ds_chirps_terciles.isel(tercile=1, drop=True)
        .where(max_index == 1)
        .sel(time=time)
        .tprate.plot.pcolormesh(
            ax=ax4, add_colorbar=False, cmap=discrete_cmap1, norm=norm1
        )
    )
    c2 = (
        ds_chirps_terciles.isel(tercile=2, drop=True)
        .where(max_index == 2)
        .sel(time=time)
        .tprate.plot.pcolormesh(
            ax=ax4, add_colorbar=False, cmap=discrete_cmap2, norm=norm2
        )
    )

    cbar0 = fig.colorbar(
        c0,
        cax=ax3,
        boundaries=bounds,
        spacing="proportional",
        orientation="horizontal",
        label="Upper",
    )
    cbar1 = fig.colorbar(
        c1,
        cax=ax2,
        boundaries=bounds,
        spacing="proportional",
        orientation="horizontal",
        label="Middle",
    )
    cbar2 = fig.colorbar(
        c2,
        cax=ax1,
        boundaries=bounds,
        spacing="proportional",
        orientation="horizontal",
        label="Lower",
    )

    cbar0.ax.xaxis.set_label_position("top")
    cbar1.ax.xaxis.set_label_position("top")
    cbar2.ax.xaxis.set_label_position("top")

    return fig


def chirps_download_and_process(country_config, geobb):
    """
    Download CHRPS precipitation data and store it as netcdf files.

    Downloads precipitation data for a given bounding box and saves the
    files as netcdf file in the `data` directory.

    Inputs
    ------
    country_config (dict): A dictionary with configuration parameters for a
        specific country. 
    geobb (GeoBoundingBox): An instance of the `GeoBoundingBox` class
        specifying the bounding box coordinates. 
    """
    chirps_monthly = ChirpsMonthly(
        country_config=country_config, geo_bounding_box=geobb
    )
    chirps_monthly.download()
    chirps_monthly.process()
    return chirps_monthly


def chirps_load(chirps_monthly):
    """
    Load CHIRPS dataset.

    Returns
    --------
    ds_chirps (xarray Dataset): CHIRPS full dataset.
    """
    ds_chirps = chirps_monthly.load()
    return ds_chirps.rename(
        {
        "X": "longitude", 
        "Y": "latitude", 
        "T": "time", 
        "precipitation": "tprate"
        }
    ).drop_vars("spatial_ref")


def chirps_create_reference(chirps_monthly_data):
    """
    Create reference dataset for CHIRPS data.

    Inputs
    ------
    chirps_monthly_data (xarray Dataset): Monthly CHIRPS dataset.

    Returns
    -------
    ds_chirps_seas_reference (xarray Dataset): Seasonal dataset obtained by 
        averaging the values of the original dataset for each three-month 
        period.
    """
    ds_chirps_seas_reference = from_monthly_to_seasonal(chirps_monthly_data)
    ds_chirps_seas_reference["tprate"] = ds_chirps_seas_reference.tprate.chunk(
        dict(time=-1)
    )
    return ds_chirps_seas_reference


def chirps_quantile_boundaries(ds_chirps_seas_reference, seasons):
    """
    Calculate the 33rd and 67th quantiles along the "time" dimension of the
    input xarray dataset for each season given in the input list. Returns an
    xarray dataset containing the quantiles for each season.
    
    Inputs
    --------
    ds_chirps_seas_reference (xarray.Dataset): Reference dataset for which the 
        quantiles need to be calculated.
    seasons (list): A list containing the name of the seasons for which the 
        quantiles need to be calculated.

    Returns
    --------
    quantiles_seas (xarray.Dataset): An xarray dataset containing the quantiles 
        along the "time" dimension for each season.
    """
    for i, seas in enumerate(seasons):
        quant_seas = (
            ds_chirps_seas_reference.where(
                ds_chirps_seas_reference["time"] == seas, drop=True
            )
            .quantile([0.33, 0.67], dim=["time"])
            .expand_dims("time")
            .assign_coords({"time": [seas]})
        )
        if i == 0:
            quantiles_seas = quant_seas.copy()
        else:
            quantiles_seas = xr.concat(
                [quantiles_seas, quant_seas], 
                dim="time"
                )
    return quantiles_seas


def chirps_tercile_probs(ds_chirps, quantiles_seas):
    """
    Calculate the probability of precipitation terciles for each season.

    This function calculates the probability of precipitation terciles (upper,
    middle, lower) for each season based on the CHIRPS dataset and the
    corresponding quantiles. The CHIRPS dataset is converted from monthly to
    seasonal data and then for each season (DJF, MAM, JJA, SON) the probability
    of precipitation terciles is calculated using the corresponding quantiles.

    Inputs
    ------
    ds_chirps (xarray.Dataset): CHIRPS dataset.
    quantiles_seas (xarray.Dataset): dataset that includes the quantiles of
        each season for the corresponding years.

    Returns
    -------
    ds_chirps_terciles (xarray.Dataset): dataset that includes the probability 
        of precipitation terciles (upper, middle, lower) for each season 
        (DJF, MAM, JJA, SON) based on the CHIRPS dataset and the corresponding 
        quantiles.
    """
    ds_chirps_seas = from_monthly_to_seasonal(ds_chirps, retain_year=True)

    for j, tercile in enumerate(["upper", "middle", "lower"]):
        for i, time in enumerate(ds_chirps_seas.time.values.tolist()):

            lower_quant_seas = quantiles_seas.sel(
                quantile=0.33, 
                drop=True
                ).sel(
                time=time.split("-")[0], drop=True
            )
            upper_quant_seas = quantiles_seas.sel(
                quantile=0.67, 
                drop=True
                ).sel(
                time=time.split("-")[0], drop=True
            )

            ds_chirps_seas_time = ds_chirps_seas.sel(time=time, drop=True)

            if tercile == "lower":
                ds_chirps_terc_time = xr.where(
                    ds_chirps_seas_time <= lower_quant_seas, 1, 0
                )
            elif tercile == "upper":
                ds_chirps_terc_time = xr.where(
                    ds_chirps_seas_time > upper_quant_seas, 1, 0
                )
            elif tercile == "middle":
                ds_chirps_terc_time = xr.where(
                    (ds_chirps_seas_time > lower_quant_seas)
                    & (ds_chirps_seas_time <= upper_quant_seas),
                    1,
                    0,
                )
            else:
                raise ValueError(f"Tercile {tercile} not defined")

            ds_chirps_terc_time = ds_chirps_terc_time.expand_dims(
                ["time", "tercile"]
            ).assign_coords({"time": [time], "tercile": [tercile]})

            if i == 0:
                ds_chirps_terc = ds_chirps_terc_time.copy()
            else:
                ds_chirps_terc = xr.concat(
                    [ds_chirps_terc, ds_chirps_terc_time], dim="time"
                )

        if j == 0:
            ds_chirps_terciles = ds_chirps_terc.copy()
        else:
            ds_chirps_terciles = xr.concat(
                [ds_chirps_terciles, ds_chirps_terc], dim="tercile"
            )

    return ds_chirps_terciles


# Iri


def iri_terciles_create(country_config, geobb, only_dominant=True):
    """
    Download and process IRI data for a given country and region.

    This function downloads and processes the IRI data for a given country and
    region, then loads it into a dataset, and finally calculates the tercile
    probabilities.

    Inputs
    --------
    country_config (dict): A dictionary containing the configuration for the 
        desired country.
    geobb (GeoBoundingBox): An object that defines the bounding box for the 
        desired region.
    only_dominant (bool): If True, only the dominant crop is used in the 
        calculations.

    Returns
    --------
    tercile_probs (xr.Dataset): A dataset containing the tercile probabilities 
        for each month.
    """
    print("Download and process data...")
    iri_monthly = iri_download_and_process(
        country_config, geobb, only_dominant=only_dominant
    )
    print("Load data...")
    ds_iri = iri_load(iri_monthly, only_dominant=only_dominant)
    return iri_tercile_probs(ds_iri, only_dominant=only_dominant)


def iri_terciles_plot(ds_iri_terciles, time):
    """
    Create a plot of the precipitation terciles for the given time using the
    IRI dataset.

    Inputs
    -------
    ds_iri_terciles (xarray Dataset): The IRI dataset.
    time (str): The time for which the plot is to be created.

    Returns
    --------
    fig (matplotlib Figure): The matplotlib figure object of the created plot.
    """
    fig, ax1, ax2, ax3, ax4 = create_fig()
    (
        norm0,
        norm1,
        norm2,
        discrete_cmap0,
        discrete_cmap1,
        discrete_cmap2,
        bounds,
    ) = create_cmaps()

    c0 = (
        ds_iri_terciles.isel(tercile=0, drop=True)
        .sel(time=time)
        .tprate.plot.pcolormesh(
            ax=ax4, add_colorbar=False, cmap=discrete_cmap0, norm=norm0
        )
    )
    c1 = (
        ds_iri_terciles.isel(tercile=1, drop=True)
        .sel(time=time)
        .tprate.plot.pcolormesh(
            ax=ax4, add_colorbar=False, cmap=discrete_cmap1, norm=norm1
        )
    )
    c2 = (
        ds_iri_terciles.isel(tercile=2, drop=True)
        .sel(time=time)
        .tprate.plot.pcolormesh(
            ax=ax4, add_colorbar=False, cmap=discrete_cmap2, norm=norm2
        )
    )

    cbar0 = fig.colorbar(
        c0,
        cax=ax3,
        boundaries=bounds,
        spacing="proportional",
        orientation="horizontal",
        label="Upper",
    )
    cbar1 = fig.colorbar(
        c1,
        cax=ax2,
        boundaries=bounds,
        spacing="proportional",
        orientation="horizontal",
        label="Middle",
    )
    cbar2 = fig.colorbar(
        c2,
        cax=ax1,
        boundaries=bounds,
        spacing="proportional",
        orientation="horizontal",
        label="Lower",
    )

    cbar0.ax.xaxis.set_label_position("top")
    cbar1.ax.xaxis.set_label_position("top")
    cbar2.ax.xaxis.set_label_position("top")

    return fig


def iri_download_and_process(country_config, geobb, only_dominant=True):
    if only_dominant:
        iri_monthly = IriForecastDominant(
            country_config=country_config, geo_bounding_box=geobb
        )
    else:
        iri_monthly = IriForecastProb(
            country_config=country_config, geo_bounding_box=geobb
        )
    iri_monthly.download()
    iri_monthly.process()
    return iri_monthly


def iri_load(iri_monthly, only_dominant=True):
    """
    Load IRI data.

    Rename the coordinates of the input dataset and return a subset of the 
    data if `only_dominant` is True, otherwise assign new coordinates and 
    return a different subset of the data.

    Inputs
    ------
    iri_monthly (xarray.Dataset): A dataset with the monthly IRI data.
    only_dominant (bool, optional): Whether to return only the dominant 
        tercile probability rates. Defaults to True.

    Returns
    -------
    ds_iri (xarray.Dataset): The input dataset with the renamed coordinates and 
        a subset of the data according to the specified conditions. 
        If only_dominant is True, then it returns the dataset with coordinates 
        longitude, latitude, time and tprate. If only_dominant is False, 
        then it returns the dataset with coordinates longitude, latitude, 
        time, tprate and tercile.

    """
    ds_iri = iri_monthly.load()
    if only_dominant:
        return (
            ds_iri.rename(
                {
            "X": "longitude", 
            "Y": "latitude", 
            "F": "time", 
            "dominant": "tprate"
            }
            )
            .drop_vars("spatial_ref")
            .sel(L=3, drop=True)
        )
    else:
        return (
            ds_iri.rename(
                {
                    "X": "longitude",
                    "Y": "latitude",
                    "F": "time",
                    "C": "tercile",
                    "prob": "tprate",
                }
            )
            .drop_vars("spatial_ref")
            .sel(L=3, drop=True)
            .assign_coords({"tercile": ["lower", "middle", "upper"]})
        )


def iri_tercile_probs(ds_iri, only_dominant=True):
    """
    Calculate the tercile probabilities of a given dataset of IRI.

    Calculate probabilities for the upper, middle, and lower terciles, based on
    the seasonal data. If only_dominant=True, return only the dominant tercile, 
    which is the one with the highest probability for each time step.

    Inputs
    --------
    ds_iri (xarray dataset): Dataset containing the IRI data. It must have the
        'tprate' variable, which is the precipitation anomaly data.
    only_dominant (bool): Flag to indicate if only the dominant tercile should
        be returned. The default is True.

    Returns
    --------
    ds_iri_terciles (xarray dataset): Dataset containing the probabilities for
        each tercile. If only_dominant=True, it only contains the dominant
        tercile for each time step.
    """
    ds_iri_seas = from_monthly_to_seasonal(
        ds_iri, 
        retain_year=True, 
        iri_data=True
        )

    if not only_dominant:
        ds_iri_seas["tprate"] = ds_iri_seas["tprate"] / 100
        return ds_iri_seas

    for j, tercile in enumerate(["upper", "middle", "lower"]):

        if tercile == "lower":
            ds_iri_terc = xr.where(
                ds_iri_seas <= -35, abs(ds_iri_seas.tprate) * 0.01, np.nan
            )
        elif tercile == "middle":
            ds_iri_terc = xr.where(
                (ds_iri_seas > -35) & (ds_iri_seas <= 35), 
                1, 
                np.nan
                )
        elif tercile == "upper":
            ds_iri_terc = xr.where(
                ds_iri_seas > 35, abs(ds_iri_seas.tprate) * 0.01, np.nan
            )
        else:
            raise ValueError(f"Tercile {tercile} not defined")

        ds_iri_terc = ds_iri_terc.expand_dims(["tercile"]).assign_coords(
            {"tercile": [tercile]}
        )

        if j == 0:
            ds_iri_terciles = ds_iri_terc.copy()
        else:
            ds_iri_terciles = xr.concat(
                [ds_iri_terciles, ds_iri_terc], 
                dim="tercile"
                )

    return ds_iri_terciles


# Ecmwf

def ecmwf_terciles_create(
    country_config,
    geobb,
    download=False,
    adapt_coordinates=False,
    ds_ref=None,
    method="linear",
):
    """
    Download and process ECMWF data to create tercile probabilities.

    Inputs
    --------
    country_config (dict): dictionary with information about the country,
        including data directories and variable names.
    geobb (tuple): tuple with the bounding box of the region of interest, in
        the format (lon_min, lat_min, lon_max, lat_max).
    download (bool): whether to download the data or use the local files.
        Default is False.
    adapt_coordinates (bool): whether to adapt the coordinates of the data to
        match the reference dataset. Default is False.
    ds_ref (xarray.Dataset): reference dataset to match the coordinates to.
        Required if adapt_coordinates is True.
    method (str): interpolation method to use when adapting the coordinates.
        Default is 'linear'.

    Returns
    --------
    tercile_probs (xarray.Dataset): tercile probabilities for the forecast 
        data.
    """
    if adapt_coordinates:
        geobb, geobb_original = increase_box(geobb)
    print("Download and process data...")
    ecmwf_download_and_process(country_config, geobb, download=download)
    print("Load data...")
    ds_ecmwf_hindcast, ds_ecmwf_forecast = ecmwf_load()
    if adapt_coordinates:
        print("Adapt coordinates...")
        if ds_ref is None:
            raise ValueError(
                "If you want to adapt coordinates you need to provide the "
                "parameter `ds_ref`."
            )
        ds_ecmwf_hindcast = coords_adapt_do_ds(
            ds=ds_ecmwf_hindcast,
            ds_ref=ds_ref,
            coord_names=["latitude", "longitude"],
            method=method,
        )
        ds_ecmwf_forecast = coords_adapt_do_ds(
            ds=ds_ecmwf_forecast,
            ds_ref=ds_ref,
            coord_names=["latitude", "longitude"],
            method=method,
        )
        ds_ecmwf_hindcast = restrict_to_geobb(
            ds_ecmwf_hindcast, 
            geobb_original
            )
        ds_ecmwf_forecast = restrict_to_geobb(
            ds_ecmwf_forecast, 
            geobb_original
            )
    print("Create climatology...")
    ds_ecmwf_seas_reference = ecmwf_create_reference(ds_ecmwf_hindcast)
    print("Calculate quantiles...")
    ecmwf_quantiles_seas = ecmwf_quantile_boundaries(
        ds_ecmwf_seas_reference, 
        seasons
        )
    print("Calculate tercile probabilities...")
    return ecmwf_tercile_probs(ds_ecmwf_forecast, ecmwf_quantiles_seas)


def ecmwf_terciles_plot(ds_ecmwf_terciles, time):
    """
    Create a pcolormesh plot of ECMWF tercile categories for a given time.

    This function creates a plot with three different color maps, one for each
    tercile category (lower, middle, and upper). The plot shows the tercile
    category with the highest probability for a given time.

    Inputs
    ------
    ds_ecmwf_terciles (xarray.Dataset): A dataset containing ECMWF tercile
        categories.
    time (str or datetime-like): A string or datetime-like object representing
        the time for which to create the plot.

    Returns
    -------
    fig (matplotlib.figure.Figure): A matplotlib figure object containing the
        tercile plot.
    """
    max_index = ds_ecmwf_terciles.argmax("tercile")

    fig, ax1, ax2, ax3, ax4 = create_fig()
    (
        norm0,
        norm1,
        norm2,
        discrete_cmap0,
        discrete_cmap1,
        discrete_cmap2,
        bounds,
    ) = create_cmaps()

    c0 = (
        ds_ecmwf_terciles.isel(tercile=0, drop=True)
        .where(max_index == 0)
        .sel(time=time)
        .tprate.plot.pcolormesh(
            ax=ax4, add_colorbar=False, cmap=discrete_cmap0, norm=norm0
        )
    )
    c1 = (
        ds_ecmwf_terciles.isel(tercile=1, drop=True)
        .where(max_index == 1)
        .sel(time=time)
        .tprate.plot.pcolormesh(
            ax=ax4, add_colorbar=False, cmap=discrete_cmap1, norm=norm1
        )
    )
    c2 = (
        ds_ecmwf_terciles.isel(tercile=2, drop=True)
        .where(max_index == 2)
        .sel(time=time)
        .tprate.plot.pcolormesh(
            ax=ax4, add_colorbar=False, cmap=discrete_cmap2, norm=norm2
        )
    )

    cbar0 = fig.colorbar(
        c0,
        cax=ax3,
        boundaries=bounds,
        spacing="proportional",
        orientation="horizontal",
        label="Upper",
    )
    cbar1 = fig.colorbar(
        c1,
        cax=ax2,
        boundaries=bounds,
        spacing="proportional",
        orientation="horizontal",
        label="Middle",
    )
    cbar2 = fig.colorbar(
        c2,
        cax=ax1,
        boundaries=bounds,
        spacing="proportional",
        orientation="horizontal",
        label="Lower",
    )

    cbar0.ax.xaxis.set_label_position("top")
    cbar1.ax.xaxis.set_label_position("top")
    cbar2.ax.xaxis.set_label_position("top")

    return fig


def ecmwf_download_and_process(country_config, geobb, download=False):
    """
    Download ECMWF forecast and hindcast prec data and store as netcdf files.

    Downloads precipitation data for a given bounding box and saves the
    hindcast and forecast files as netcdf files in the `data` directory.

    Inputs
    ------
    country_config (dict): A dictionary with configuration parameters for a
        specific country. 
    geobb (GeoBoundingBox): An instance of the `GeoBoundingBox` class
        specifying the bounding box coordinates. 
    download (bool): Flag indicating whether to download the data or not.
    """
    if download:

        area = [geobb.lat_max, geobb.lon_min, geobb.lat_min, geobb.lon_max]

        c = cdsapi.Client()

        data_request_netcdf = {
            "format": "netcdf",
            "originating_centre": "ecmwf",
            "system": "5",
            "variable": "total_precipitation",
            "product_type": "monthly_mean",
            "year": [f"{d}" for d in range(1993, 2017)],
            "month": [f"{d:02d}" for d in range(1, 13)],
            "leadtime_month": "3",
            "area": area,
        }

        c.retrieve(
            "seasonal-monthly-single-levels",
            data_request_netcdf,
            f"data/ecmwf-hindcast-total.nc",
        )

        data_request_netcdf = {
            "format": "netcdf",
            "originating_centre": "ecmwf",
            "system": "5",
            "variable": "total_precipitation",
            "product_type": "monthly_mean",
            "year": [f"{d}" for d in range(1981, 2023)],
            "month": [f"{d:02d}" for d in range(1, 13)],
            "leadtime_month": "3",
            "area": area,
        }

        c.retrieve(
            "seasonal-monthly-single-levels",
            data_request_netcdf,
            f"data/ecmwf-forecast-total.nc",
        )


def ecmwf_load():
    """
    Load ECMWF hindcast and forecast datasets.

    Loads two NetCDF datasets: one for ECMWF hindcast and another for forecast.

    Returns
    --------
    ds_ecmwf_hindcast (xarray Dataset): ECMWF hindcast dataset.
    ds_ecmwf_forecast (xarray Dataset): ECMWF forecast dataset.
    """
    ds_ecmwf_hindcast = xr.open_dataset("data/ecmwf-hindcast-total.nc")
    ds_ecmwf_forecast = xr.open_dataset("data/ecmwf-forecast-total.nc")
    return ds_ecmwf_hindcast, ds_ecmwf_forecast


def ecmwf_create_reference(ds_ecmwf_hindcast):
    """
    Create reference dataset for ECMWF data.

    This function takes a dataset of ECMWF hindcast with monthly data and
    converts it to a seasonal dataset by averaging the values of each
    three months.

    Inputs
    ------
    ds_ecmwf_hindcast (xarray Dataset): Monthly ECMWF hindcast dataset.

    Returns
    -------
    ds_seasonal (xarray Dataset): Seasonal dataset obtained by averaging
        the values of the original dataset for each three-month period.
    """
    return from_monthly_to_seasonal(ds_ecmwf_hindcast)


def ecmwf_quantile_boundaries(ds_ecmwf_seas_reference, seasons):
    """
    Compute the quantile boundaries of a dataset for a set of seasons.

    Inputs
    --------
    ds_ecmwf_seas_reference (xarray Dataset): Reference dataset containing
        the seasonal values.
    seasons (list of str): Seasons for which to compute the quantile
        boundaries.

    Returns
    --------
    quantiles_seas (xarray Dataset): Dataset containing the quantile
        boundaries for the specified seasons.
    """
    for i, seas in enumerate(seasons):
        quant_seas = (
            ds_ecmwf_seas_reference.where(
                ds_ecmwf_seas_reference["time"] == seas, drop=True
            )
            .quantile([0.33, 0.67], dim=["time", "number"])
            .expand_dims("time")
            .assign_coords({"time": [seas]})
        )
        if i == 0:
            quantiles_seas = quant_seas.copy()
        else:
            quantiles_seas = xr.concat(
                [quantiles_seas, quant_seas], 
                dim="time"
                )
    return quantiles_seas


def ecmwf_tercile_probs(ds_ecmwf_forecast, quantiles_seas):
    """
    Calculate the probability of ECMWF forecasts falling within each tercile.

    This function takes an ECMWF forecast data and a quantile data as inputs,
    and calculates the probability of the forecast falling within each tercile
    (lower, middle, and upper) for each season.

    Inputs
    --------
    ds_ecmwf_forecast (xarray.Dataset): A dataset of ECMWF forecast data.
    quantiles_seas (xarray.Dataset): A dataset of seasonally stratified
        quantile data.

    Returns
    --------
    ds_ecmwf_terciles (xarray.Dataset): A dataset of the probability of the
        forecast falling within each tercile (lower, middle, and upper) for 
        each season.
    """
    ds_ecmwf_seas = from_monthly_to_seasonal(
        ds_ecmwf_forecast, 
        retain_year=True
        )

    number_list = list(range(51))
    number_array = xr.DataArray(number_list, dims=["number"])

    for j, tercile in enumerate(["upper", "middle", "lower"]):

        for i, time in enumerate(ds_ecmwf_seas.time.values.tolist()):

            lower_quant_seas = quantiles_seas.sel(
                quantile=0.33, drop=True).sel(
                time=time.split("-")[0], drop=True
            )
            lower_quant_seas_tiled = xr.concat(
                [lower_quant_seas] * len(number_array), dim="number"
            ).assign_coords({"number": list(range(51))})
            upper_quant_seas = quantiles_seas.sel(
                quantile=0.67, drop=True
                ).sel(
                time=time.split("-")[0], drop=True
            )
            upper_quant_seas_tiled = xr.concat(
                [upper_quant_seas] * len(number_array), dim="number"
            ).assign_coords({"number": list(range(51))})

            ds_ecmwf_seas_time = ds_ecmwf_seas.sel(time=time, drop=True)

            ds_ecmwf_seas_time_nonan = ds_ecmwf_seas_time.count(dim="number")

            if tercile == "lower":
                ds_temp = (ds_ecmwf_seas_time <= lower_quant_seas_tiled).sum(
                    dim="number"
                )
            elif tercile == "upper":
                ds_temp = (ds_ecmwf_seas_time > upper_quant_seas_tiled).sum(
                    dim="number"
                )
            elif tercile == "middle":
                ds_temp = (
                    (ds_ecmwf_seas_time > lower_quant_seas_tiled)
                    & (ds_ecmwf_seas_time <= upper_quant_seas_tiled)
                ).sum(dim="number")
            else:
                raise ValueError(f"Tercile {tercile} not defined")

            ds_ecmwf_terc_time = (
                xr.where(
                    ds_ecmwf_seas_time_nonan == 0,
                    np.nan,
                    ds_temp / ds_ecmwf_seas_time_nonan,
                )
                .expand_dims(["time", "tercile"])
                .assign_coords({"time": [time], "tercile": [tercile]})
            )

            if i == 0:
                ds_ecmwf_terc = ds_ecmwf_terc_time.copy()
            else:
                ds_ecmwf_terc = xr.concat(
                    [ds_ecmwf_terc, ds_ecmwf_terc_time], dim="time"
                )

        if j == 0:
            ds_ecmwf_terciles = ds_ecmwf_terc.copy()
        else:
            ds_ecmwf_terciles = xr.concat(
                [ds_ecmwf_terciles, ds_ecmwf_terc], dim="tercile"
            )

    return ds_ecmwf_terciles