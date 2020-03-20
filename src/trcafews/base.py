# -*- coding: utf-8 -*-
"""
This is the base file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         epaswmm = trcafews.base:run

Then run `python setup.py install` which will install the command `pyhypack`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

"""

import argparse as ap
import sys
import datetime
import math
import os
import shutil
import string
import subprocess
import sys
import xml
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom.minidom import parse

import trcafews
import numpy as np
import pandas as pd
import xarray as xr

from trcafews import __version__

__author__ = "pbishop,lboutin"
__copyright__ = "pbishop,lboutin"
__license__ = "mit"


# XML namespace dict, needed to find elements
namespace = {"pi": "http://www.wldelft.nl/fews/PI"}

# TODO read these from a command line or configuration file
# the exportDir where the run_info.xml and input files from Delft-FEWS can be found
fews_export_dir = Path("FromFewsToModel")
run_info_file = fews_export_dir / "run_info.xml"
# determine how we get the timestep, for now impose 15 minutes.
timestep = datetime.timedelta(minutes=15)

# the tmp dir is needed to put modified metaswap input files
# that the model will copy to the output folder
# if we put them in the output folder directly, metaswap
# cannot handle this
tmp_dir = Path("tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)
# put these in the parent because otherwise metaswap will get
# confused when copying the files to the output directory
# at the beginning of the run
mete_grid_file = tmp_dir / "mete_grid.inp"
para_sim_dst = tmp_dir / "para_sim.inp"

# XML namespace dict, needed to find elements
namespace = {"pi": "http://www.wldelft.nl/fews/PI"}

# constants for the water balance
mf_balance_components = ["bdgdrn", "bdgriv", "bdgisg"]
ms_balance_components = ["bdgqrun"]

# names of the water balance areas, used as long_name in the NetCDF output
station_names = [
    "Zeijen",
    "Ubbena",
    "Peesterweg",
    "Zeijerlaar",
    "Zwartedijk",
    "Donderen",
]
long_name = {
    "bdgdrn_sys1": "afvoer buisdrainage",
    "bdgdrn_sys1_neg": "afvoer buisdrainage",
    "bdgdrn_sys1_pos": "N/A",
    "bdgdrn_sys2": "afvoer greppeldrainage",
    "bdgdrn_sys2_neg": "afvoer greppeldrainage",
    "bdgdrn_sys2_pos": "N/A",
    "bdgisg_sys1": "oppervlaktewater Hunze en Aa's legger",
    "bdgisg_sys1_neg": "drainage oppervlaktewater Hunze en Aa's legger",
    "bdgisg_sys1_pos": "toevoer oppervlaktewater Hunze en Aa's legger",
    "bdgisg_sys2": "oppervlaktewater Noorderzijlvest legger",
    "bdgisg_sys2_neg": "drainage oppervlaktewater Noorderzijlvest legger",
    "bdgisg_sys2_pos": "toevoer oppervlaktewater Noorderzijlvest legger",
    "bdgisg_sys3": "oppervlaktewater Hunze en Aa's TOP10 lijn",
    "bdgisg_sys3_neg": "drainage oppervlaktewater Hunze en Aa's TOP10 lijn",
    "bdgisg_sys3_pos": "toevoer oppervlaktewater Hunze en Aa's TOP10 lijn",
    "bdgisg_sys4": "oppervlaktewater Noorderzijlvest TOP10 lijn",
    "bdgisg_sys4_neg": "drainage oppervlaktewater Noorderzijlvest TOP10 lijn",
    "bdgisg_sys4_pos": "toevoer oppervlaktewater Noorderzijlvest TOP10 lijn",
    "bdgqrun": "maaiveld afvoer",
    "bdgqrun_neg": "maaiveld afvoer",
    "bdgqrun_pos": "N/A",
    "bdgriv_sys1": "oppervlaktewater TOP10 vlak",
    "bdgriv_sys1_neg": "drainage oppervlaktewater TOP10 vlak",
    "bdgriv_sys1_pos": "toevoer oppervlaktewater TOP10 vlak",
}


def grid_definition(runfile_dir):
    prj2run_template = runfile_dir / "RUNFILE_TEMPLATE.INI"
    with open(prj2run_template) as f:
        for line in f:
            k, v = line.strip().split("=")
            if k.lower() == "window":
                window = list(map(float, v.split(",")))
                xmin, ymin, xmax, ymax = window
            if k.lower() == "cellsize":
                cellsize = float(v)
    return xmin, ymin, xmax, ymax, cellsize


# read XML
def read_run_info(run_info_file):
    run_info = {}
    tree = ET.parse(run_info_file)
    root = tree.getroot()

    st = time_element(root.find("pi:startDateTime", namespace))
    et = time_element(root.find("pi:endDateTime", namespace))

    # bad but effictive way to deal with timezone hour differences
    run_info["start_time"] = pd.Timestamp(st).round("1d")
    run_info["end_time"] = pd.Timestamp(et).round("1d")

    # we use the same working directory as Deft-FEWS, to keep relative paths easy
    run_info["work_dir"] = dir_element(root.find("pi:workDir", namespace))
    os.chdir(run_info["work_dir"])

    run_info["statefiles"] = [
        file_element(e) for e in root.findall("pi:inputStateDescriptionFile", namespace)
    ]
    run_info["mapstacks"] = [
        file_element(e) for e in root.findall("pi:inputMapStackFile", namespace)
    ]
    run_info["diagnostic_xml"] = file_element(
        root.find("pi:outputDiagnosticFile", namespace), exists=False
    )

    # to keep the number of configuration files to a minimum,
    # we put extra properties in the run_info.xml
    properties = root.find("pi:properties", namespace)
    run_info["properties"] = {}
    for e in properties.findall("pi:string", namespace):
        key = e.get("key")
        val = e.get("value")
        if key == "output-directory":
            path = dir_element(val, exists=False)
            path.mkdir(parents=True, exist_ok=True)
        elif key.endswith("-directory"):
            path = dir_element(val)
        elif key.startswith("output-"):
            path = file_element(val, exists=False)
        else:
            path = file_element(val)
        run_info["properties"][key] = path
    # check these here for better error messages
    check_properties("imod-executable", run_info["properties"], run_info_file)
    check_properties("model-executable", run_info["properties"], run_info_file)
    check_properties("project-file", run_info["properties"], run_info_file)
    check_properties("state-directory", run_info["properties"], run_info_file)
    check_properties("output-directory", run_info["properties"], run_info_file)

    return run_info


def time_element(elem):
    """Get datetime from XML element with date and time attributes"""
    start_str = elem.get("date") + " " + elem.get("time")
    return datetime.datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")


def file_element(elem, exists=True):
    if isinstance(elem, str):
        # such that this works if the path is in an attribute
        path = Path(elem)
    else:
        path = Path(elem.text)
    if exists and not path.is_file():
        raise FileNotFoundError(path.resolve())
    return path


def dir_element(elem, exists=True):
    if isinstance(elem, str):
        # such that this works if the path is in an attribute
        path = Path(elem)
    else:
        path = Path(elem.text)
    if exists and not path.is_dir():
        raise FileNotFoundError(path.resolve())
    return path


def check_properties(key, properties, run_info_file):
    if key not in properties:
        raise KeyError(
            f'"{key}" needs to be specified under <properties> in {run_info_file.resolve()}'
        )


def fix_column(line):
    parts = line.split(",")
    del parts[1]
    return ",".join(parts)


def is_cap_header(line):
    # works for both PRJ and RUN
    parts = line.split(",", maxsplit=2)
    if len(parts) >= 2:
        if parts[1].strip().startswith("(CAP)"):
            return True
    return False


def path_from_prj(filename, prj_file):
    """Get the full path of a filename from the project file"""
    with open(prj_file) as f:
        for line in f:
            filename = filename.lower()
            strippedline = line.rstrip(string.whitespace + "'\"").lower()
            if strippedline.endswith(filename):
                path = line.rstrip().rsplit(",", maxsplit=1)[-1].strip("'\"")
                return file_element(path)


def cap_nlines(prj_file):
    found_cap_header = False
    linenr_cap_header = -9999
    nbase = -9999
    nextra = -9999
    with open(prj_file) as f:
        for linenr, line in enumerate(f):
            if is_cap_header(line):
                found_cap_header = True
                linenr_cap_header = linenr
                continue
            if found_cap_header:
                nbase = int(line.split(",")[0])
                found_cap_header = False
                continue
            if linenr == linenr_cap_header + nbase + 2:
                nextra = int(line.split(",")[0])
                break

    return nbase, nextra


def fix_cap(run_lines, nbase, nextra):
    """Modifies the run_lines inplace to fix the CAP package

    Fixes the number of lines and removes a column of ones."""

    # find the CAP header line with the count
    has_cap = False
    ncap = nbase + nextra
    for i in range(len(run_lines)):
        line = run_lines[i]
        if is_cap_header(line):
            has_cap = True
            break

    if has_cap:
        # fix reference to mete_grid.inp and para_sim.inp
        for linenr in range(i + nbase + 1, i + ncap + 1):
            path = Path(run_lines[linenr])
            if path.name.lower() == "mete_grid.inp":
                run_lines[linenr] = " " + str(mete_grid_file.resolve())
            elif path.name.lower() == "para_sim.inp":
                run_lines[linenr] = " " + str(para_sim_dst.resolve())

    return run_lines


def remove_empty_coldstate(run_lines, is_cold):
    if is_cold:
        run_lines_fixed = []
        nremoved = 0
        for l in run_lines:
            ls = l.rstrip()
            if ls.endswith("init_svatvg.inp") or ls.endswith("init_svattemp.inp"):
                nremoved += 1
            else:
                run_lines_fixed.append(l)
        return run_lines_fixed, nremoved
    else:
        # no a coldstate, nothin
        return run_lines, 0


def write_runfile(run_info):
    properties = run_info["properties"]
    imod_bin = properties["imod-executable"]
    prj_file = properties["project-file"]
    output_dir = properties["output-directory"]
    run_file = prj_file.with_suffix(".run")
    prj2run_template = prj_file.with_name("RUNFILE_TEMPLATE.INI")
    prj2run_file = prj_file.with_name("RUNFILE.INI")
    # we check on "init_svatvg.inp", but could have equally chosen "init_svattemp.inp"
    is_cold = state_is_cold(run_info["statefiles"], "init_svatvg.inp")

    sdate = run_info["start_time"].strftime("%Y%m%d%H%M%S")
    edate = run_info["end_time"].strftime("%Y%m%d%H%M%S")

    with open(prj2run_template) as fr:
        with open(prj2run_file, "w") as fw:
            for line in fr:
                key, val = line.strip().split("=")
                if key == "PRJFILE_IN":
                    val = str(prj_file)
                elif key == "RUNFILE_OUT":
                    val = str(run_file)
                elif key == "SDATE":
                    val = sdate
                elif key == "EDATE":
                    val = edate
                fw.write(key + "=" + val + "\n")

    # run iMOD batch RUNFILE function to create runfile
    subprocess.run([str(imod_bin), str(prj2run_file)], check=True)

    # read contents of runfile for manual tweaking
    with open(run_file) as f:
        run = f.read()

    # TODO fix in iMOD
    run_lines = run.splitlines()
    nbase, nextra = cap_nlines(prj_file)
    run_lines, nremoved = remove_empty_coldstate(run_lines, is_cold=is_cold)
    run_lines = fix_cap(run_lines, nbase, nextra - nremoved)

    # this replaces the relative path to the model directory with an absolute path
    # TODO check if this is really needed, because it makes the whole less portable
    run_lines[0] = f"'{output_dir.resolve()}'"

    if run_lines[3].endswith(",1"):  # extra ",1" at end of line 4
        run_lines[3] = run_lines[3][:-2]
    run_fixed = "\n".join(run_lines) + "\n"

    # write corrected runfile back to disk
    with open(run_file, "w") as f:
        f.write(run_fixed)


def mete_grid_line(td, iy, prec, etref, tempmn, tempmx, temp, nrel, rad, hum, wind):
    line = f"{td},{iy},"
    # these need to be wrapped in double quotes
    datacols = [prec, etref, tempmn, tempmx, temp, nrel, rad, hum, wind]
    dataline = '","'.join(map(str, datacols))
    line += f'"{dataline}"\n'
    return line


def day_year(t):
    day = datetime.timedelta(days=1)
    return (t - datetime.datetime(t.year, 1, 1)) / day


# convert scalars to rasters since metaswap doesn't accept scalars
def asc(parameter, time, val, griddef):
    xmin, ymin, xmax, ymax, cellsize = griddef
    timestr = time.strftime("%Y%m%d")
    path = fews_export_dir / "asc" / f"{parameter}_{timestr}.asc"
    path = path.resolve()
    (fews_export_dir / "asc").mkdir(parents=True, exist_ok=True)
    nrow = round((ymax - ymin) / cellsize)
    ncol = round((xmax - xmin) / cellsize)
    shape = (nrow, ncol)
    dims = ("y", "x")
    x = np.arange(xmin, xmax, cellsize) + 0.5 * cellsize
    y = np.arange(ymax, ymin, -cellsize) + 0.5 * -cellsize
    coords = {"y": y, "x": x}
    da = xr.DataArray(np.full((shape), val), coords=coords, dims=dims)
    da.attrs["SIGNIFICANT_DIGITS"] = 6
    imod.rasterio.write(path, da)
    return path


def prepare_grid_dataarray(input_grid_file):
    ds = xr.open_dataset(input_grid_file)
    da = ds["precipitation"]

    # ensure the y axis is decreasing
    y_decreasing = da.indexes["y"].is_monotonic_decreasing
    if not y_decreasing:
        da = da.reindex(y=da.y[::-1], copy=False)

    # effectively move the timestamp to midnight if it is not
    da = da.resample(time="1D").nearest(tolerance="2H")
    return da


def write_mete_grid(
        mete_grid_file, times, input_grid_file, input_scalar_file, runfile_dir
):
    ds_all = xr.open_dataset(input_scalar_file)
    ds1d = ds_all.squeeze("stations")
    ds = ds1d.drop(
        ["analysis_time", "lat", "lon", "y", "x", "z", "station_id", "station_names"]
    )
    df_dt = ds.to_dataframe()
    df = df_dt.resample("D").first()

    griddef = grid_definition(runfile_dir)

    da_precipitation = prepare_grid_dataarray(input_grid_file)
    # da_precipitation[...] = 0.8  # TODO remove

    # write mete_grid.inp
    with open(mete_grid_file, "w") as f:
        for t in times:
            # should be a float to support smaller timesteps
            day_of_year = day_year(t)
            if t == times[-1]:
                # this is the extra timestep needed for metaswap
                # reuse the previous day
                prev_day = t - pd.Timedelta(1, "D")
                row = df.loc[prev_day]
                prec = da_precipitation.sel(time=prev_day)
            else:
                row = df.loc[t]
                prec = da_precipitation.sel(time=t)

            timestr = t.strftime("%Y%m%d")
            prec_path = fews_export_dir / "asc" / f"precipitation_{timestr}.asc"
            imod.rasterio.write(prec_path, prec)

            line = mete_grid_line(
                day_of_year,
                t.year,
                prec=prec_path.resolve(),
                etref=asc("evaporation", t, row["evaporation"], griddef),
                tempmn="NoValue",
                tempmx="NoValue",
                temp="NoValue",
                nrel="NoValue",
                rad="NoValue",
                hum="NoValue",
                wind="NoValue",
            )
            f.write(line)


def edit_para_sim(para_sim_file, para_sim_dst, start_time):
    shutil.copy2(para_sim_file, para_sim_dst)

    with open(para_sim_file) as f:
        with open(para_sim_dst, "w") as fw:
            for line in f:
                # copy commment and empty lines
                if line.strip().startswith("*") or line.strip() == "":
                    fw.write(line)
                else:
                    # copy key = value lines, modify if key is "idbg" or "iybg"
                    key, _ = line.split("=", 1)
                    key_stripped = key.strip().lower()
                    if key_stripped == "idbg":
                        idbg = day_year(start_time)
                        fw.write(f"{key}= {idbg}\n")
                    elif key_stripped == "iybg":
                        iybg = start_time.year
                        fw.write(f"{key}= {iybg}\n")
                    else:
                        fw.write(line)


def state_is_cold(statefiles, filename):
    is_cold = True
    for statefile in statefiles:
        tree = ET.parse(statefile)
        root = tree.getroot()
        statelocs = root.findall("pi:stateLoc", namespace)
        for stateloc in statelocs:
            readloc = file_element(stateloc.find("pi:readLocation", namespace))
            # TODO check what read and write here are supposed to mean
            # TODO fix in iMOD: add the location of this file in the PRJ file
            if readloc.name == filename:
                # TODO in WOFOST, a timestep independent cold state can only be
                # done by not putting the state file. We work around this by using
                # a 0 byte "init_svatvg.inp" as a FEWS cold state, and if the state
                # is 0 bytes, we do not put it in the RUN file
                if readloc.stat().st_size != 0:
                    is_cold = False
    return is_cold


def pre_adapter():
    print("Running iMOD Delft-FEWS pre adapter")
    run_info = read_run_info(run_info_file)

    # get para_sim path from project file
    prj_file = run_info["properties"]["project-file"]
    para_sim_file = path_from_prj("para_sim.inp", prj_file)
    if para_sim_file is None:
        raise AssertionError(f"'para_sim.inp' not found in {prj_file.resolve()}")

    write_runfile(run_info)

    timespan = run_info["end_time"] - run_info["start_time"]
    # return the ceiling to make sure end_time is always simulated
    ntimestep = math.ceil(timespan / timestep)
    # these are the start times of all timesteps
    times = [run_info["start_time"] + x * timestep for x in range(ntimestep)]
    # from metaswap manual:
    # There should be entries at least up till the time that the run ends,
    # i.e. with at least one entry on or beyond the ending time of the simulation.
    times_metaswap = times + [run_info["start_time"] + ntimestep * timestep]

    input_grid_file = Path(run_info["properties"]["input-grid-file"])
    input_scalar_file = Path(run_info["properties"]["input-scalar-file"])
    assert input_grid_file.is_file()
    assert input_scalar_file.is_file()

    write_mete_grid(
        mete_grid_file,
        times_metaswap,
        input_grid_file,
        input_scalar_file,
        prj_file.parent,
    )
    edit_para_sim(para_sim_file, para_sim_dst, run_info["start_time"])


def pretty_xml(xmlpath):
    """Replaces an XML file with a pretty printed version.
    Based on http://stackoverflow.com/a/1206856, but stripping blank lines.
    """
    x = xml.dom.minidom.parse(str(xmlpath))
    xstr = x.toprettyxml(encoding="utf-8")
    lines = xstr.splitlines()
    with open(xmlpath, "wb") as f:
        for line in lines:
            if line.strip() != b"":
                f.write(line + b"\n")


def run_model():
    print("Running iMOD model")
    run_info = read_run_info(run_info_file)
    prj_file = run_info["properties"]["project-file"]
    run_file = prj_file.with_suffix(".run")
    model_bin = run_info["properties"]["model-executable"]
    subprocess.run([str(model_bin), str(run_file)], check=True)


def rename_sys_idfs(output_dir):
    """rename _sys in filename with -sys to interpret it as a separate DataArray"""
    idf_sys_paths = list(output_dir.glob("**/*_sys*.idf"))
    for idf_sys_path in idf_sys_paths:
        idfsys_path = idf_sys_path.with_name(idf_sys_path.name.replace("_sys", "-sys"))
        idf_sys_path.replace(idfsys_path)


def prepare_metaswap_area(like, wet_area_path):
    """Prepare a DataArray of the metaswap area per cell in m2"""

    wet_area = imod.idf.open(wet_area_path)
    full_cell_area = wet_area.dx * -wet_area.dy

    # fix possible errors in the grid
    wet_area_clip = wet_area.clip(0.0, full_cell_area)
    regridder = imod.prepare.Regridder(method="conductance")
    wet_area_domain = regridder.regrid(wet_area_clip, like)
    # metaswap area is what's not wet
    metaswap_area = (like.dx * -like.dy) - wet_area_domain
    return metaswap_area


def group_zones(
        output_dir, mf_balance_components, ms_balance_components, metaswap_area, zones
):
    """"Load selected budget output files and groupby water balance zones"""
    dd_grid = {}
    for balance_component in mf_balance_components:
        d = imod.idf.open_dataset(
            output_dir / balance_component / f"{balance_component}*.idf"
        )
        dd_grid.update(d)
    for balance_component in ms_balance_components:
        # metaswap in in a separate output subfolder
        # and has no systems, so no need for open_dataset
        da_mm = imod.idf.open(
            output_dir / "metaswap" / balance_component / f"{balance_component}*.idf"
        )
        # metaswap is in mm/day, convert to m3/day
        da_m3 = da_mm * metaswap_area * 0.001
        # regrid metaswap to modflow extent, effectively adding an outer border of cells
        regridder = imod.prepare.Regridder(method="mean")
        da_m3_mfgrid = regridder.regrid(da_m3, zones)
        # fill with zeros such that groupby doesn't get nans
        dd_grid.update({balance_component: da_m3_mfgrid.fillna(0.0)})

    # next to the default net term, also split the cellwise positive and negatives
    # such that they can be tracked separately
    dd_grid_split = dd_grid.copy()
    for name, da in dd_grid.items():
        da = da.fillna(0.0)
        dd_grid_split[name + "-pos"] = da.where(da > 0.0, other=0.0)
        dd_grid_split[name + "-neg"] = da.where(da < 0.0, other=0.0)

    # since the dict is / can be jagged in the layers, and we need them together in a xr.Dataset
    # to be able to aggregate by zones, we should groupby individually, such that they have the
    # same (time, zones) dimensions and they can be combined in a xr.Dataset like that
    # astype(np.int) changes nan to -2147483648, which we must filter out later
    ds_zones = zones.astype(np.int).to_dataset()
    dd_zones = {}
    for name, da in dd_grid_split.items():
        name = name.replace("-", "_")  # avoid dashes in variable names
        # prepare a xr.Dataset with just this variable and the zones
        ds = ds_zones.copy()
        ds[name] = da
        # group it and store in a dict
        da_grouped = ds.groupby("station_id").sum().sum(dim="layer")[name].load()
        dd_zones[name] = da_grouped.assign_attrs(
            units="m3", long_name=long_name[name], coordinates="station_name"
        )

    ds = xr.Dataset(dd_zones).drop(["dx", "dy"]).fillna(0.0)
    # filter out the old nan values
    ds = ds.isel(station_id=slice(1, None))

    # check if stations_names is applicable
    assert np.allclose(ds.station_id, np.arange(1, 7))
    ds["station_name"] = xr.DataArray(
        station_names, coords=[ds.station_id], dims=["station_id"], name="station_name",
    )
    return ds


def add_attributes(ds):
    """Add more attributes to make it more CF compliant"""
    ds.time.attrs["standard_name"] = "time"
    ds.time.attrs["axis"] = "T"
    ds.station_id.attrs["cf_role"] = "timeseries_id"

    ds = ds.assign_attrs(
        Conventions="CF-1.6",
        title="Water balance",
        institution="Deltares",
        references="http://www.delft-fews.com",
        Metadata_Conventions="Unidata Dataset Discovery v1.0",
        summary="Modflow-Metaswap water balance output",
        date_created=datetime.datetime.utcnow().replace(microsecond=0).isoformat(" ")
                     + " GMT",
        coordinate_system="Rijks Driehoekstelsel",
        featureType="timeSeries",
    )

    return ds


def waterbalance(run_info):
    """
    Run a water balance, grouping budget files per water balance zone.
    Outputs a NetCDF that can be read into Delft-FEWS.
    """
    # get the needed settings from run_info
    properties = run_info["properties"]
    output_dir = properties["output-directory"]
    wbal_zones_path = properties["wbal-zones-file"]
    wet_area_path = properties["wet-area-file"]
    waterbalance_path = properties[
        "output-waterbalance-file"
    ]  # value="%WORK_DIR%/FromModelToFews\waterbalance.nc"/>

    # load the zoning grid which we use for grouping
    zones = imod.idf.open(wbal_zones_path).load()
    zones.name = "station_id"

    # rename files such that the imod-python naming conventions work
    rename_sys_idfs(output_dir)
    # calculate metaswap_area to be able to convert from mm to m3
    metaswap_area = prepare_metaswap_area(like=zones, wet_area_path=wet_area_path)

    # load selected budget output files and group them by the water balance zones
    ds = group_zones(
        output_dir, mf_balance_components, ms_balance_components, metaswap_area, zones
    )

    # write a NetCDF with the right metadata such that it can be read into Delft-FEWS
    ds = add_attributes(ds)
    ds.to_netcdf(waterbalance_path)


def write_ts_netcdf(run_info):
    """
    Save a subset of the model output grids as a NetCDF, to be imported into Delft-FEWS.
    """
    properties = run_info["properties"]
    output_dir = properties["output-directory"]

    # load the most interesting grid data
    #head = imod.idf.open(output_dir / "head" / "head_*_l1.idf")
    #trel = imod.idf.open(output_dir / "metaswap" / "msw_Trel" / "msw_Trel_*_L1.IDF")

    # take the single layer out of the dimensions
    #head = head.squeeze("layer", drop=True)
    #trel = trel.squeeze("layer", drop=True)
    # trel is one outer ring smaller, reindex it on the modflow grid, introducing nans
    #trel = trel.reindex_like(head)

    # combine them in a Dataset
    ds = head.to_dataset()
    ds["trel"] = trel

    # add metadata attributes to make it more CF compliant
    ds = ds.assign_attrs(
        Conventions="CF-1.6",
        title="Model gridded output",
        institution="Deltares",
        references="http://www.delft-fews.com",
        Metadata_Conventions="Unidata Dataset Discovery v1.0",
        summary="Modflow-Metaswap gridded model output",
        date_created=datetime.datetime.utcnow().replace(microsecond=0).isoformat(" ")
                     + " GMT",
        coordinate_system="Rijks Driehoekstelsel",
    )

    ds["head"] = ds["head"].assign_attrs(units="m", long_name="grondwaterstand")
    ds["trel"] = ds["trel"].assign_attrs(units="m", long_name="relatieve transpiratie")
    ds["x"] = ds["x"].assign_attrs(
        units="m",
        long_name="x coordinate",
        standard_name="projection_x_coordinate",
        axis="X",
    )
    ds["y"] = ds["y"].assign_attrs(
        units="m",
        long_name="y coordinate",
        standard_name="projection_y_coordinate",
        axis="Y",
    )
    ds["time"] = ds["time"].assign_attrs(standard_name="time", axis="T")
    ds["dx"] = ds["dx"].assign_attrs(units="m", long_name="cell size along x dimension")
    ds["dy"] = ds["dy"].assign_attrs(units="m", long_name="cell size along y dimension")

    # forecast reference time for Delft-FEWS
    # https://publicwiki.deltares.nl/display/FEWSDOC/NetCDF+formats+that+can+be+imported+in+Delft-FEWS
    ds = ds.expand_dims({"analysis_time": 1})
    ds = ds.assign_coords({"analysis_time": [np.datetime64("now", "ms")]})
    ds["analysis_time"] = ds["analysis_time"].assign_attrs(
        standard_name="forecast_reference_time"
    )

    ds.to_netcdf(grid_path)


def post_adapter():
    print("Running post adapter")
    run_info = read_run_info(run_info_file)
    properties = run_info["properties"]
    statefiles = run_info["statefiles"]
    end_time = run_info["end_time"]
    states_dir = properties["state-directory"]
    output_dir = properties["output-directory"]
    config_out_dir = states_dir / "config_out"
    config_out_dir.mkdir(parents=True, exist_ok=True)
    files_out_dir = states_dir / "files_out"
    files_out_dir.mkdir(parents=True, exist_ok=True)

    # copy statefiles to config_out
    for statefile_in in statefiles:
        statefilename_out = statefile_in.name.replace("_in.xml", "_out.xml")
        statefile_out = config_out_dir / statefilename_out

        ET.register_namespace("", namespace["pi"])
        tree = ET.parse(statefile_in)
        root = tree.getroot()

        # set the end date from the run_info
        elem_datetime = root.find("pi:dateTime", namespace)
        elem_datetime.set("date", end_time.strftime("%Y-%m-%d"))
        elem_datetime.set("time", end_time.strftime("%H:%M:%S"))
        elem_datetime = root.find("pi:dateTime", namespace)

        # copy and rename the output files to the states/files_out directory
        statelocs = root.findall("pi:stateLoc", namespace)
        for stateloc in statelocs:
            read_elem = stateloc.find("pi:readLocation", namespace)
            write_elem = stateloc.find("pi:writeLocation", namespace)
            readloc = file_element(read_elem)
            writeloc = file_element(write_elem)
            # the "files_in" directory is changed to "files_out"
            read_elem.text = str(readloc.parents[1] / "files_out" / readloc.name)
            write_elem.text = str(writeloc.parents[1] / "files_out" / writeloc.name)
            if readloc.stem.startswith("init_svat"):
                # handles init_svat/init_svatvg/init_svattemp
                shutil.copy2(
                    output_dir / "metaswap" / (readloc.stem + ".out"),
                    files_out_dir / (readloc.stem + ".inp"),
                )
            elif readloc.stem.startswith("sh_"):
                # handles modflow heads
                head = imod.idf.open(output_dir / "head" / "head_*.idf")
                # take the last timestep, and drop the coordinate to prevent
                # it from appearing in the filename when saving
                head_last = head.isel(time=-1).drop("time")
                imod.idf.save(files_out_dir / "sh", head_last)

        # write the modified XML to a file
        tree.write(statefile_out)
        pretty_xml(statefile_out)

    # extract timeseries from output files either *.rpt or *.out
    extract_timeseries(run_info)
    # save model output as a NetCDF,
    # to be imported into Delft-FEWS
    write_ts_netcdf(run_info)


## parse the command line arguments

# execute only if run as a script
if __name__ == "__main__":
    # TODO add help text
    parser = argparse.ArgumentParser(description="TRCA-FEWS adapter for EPASWMM models")
    parser.set_defaults(func=parser.print_usage)

    subparsers = parser.add_subparsers(
        title="subcommands", description="valid subcommands", help="additional help"
    )

    # create the parser for the "pre" command
    help_pre = "Prepare the EPASWMM model"
    parser_pre = subparsers.add_parser("pre", help=help_pre)
    parser_pre.set_defaults(func=pre_adapter)

    # create the parser for the "run" command
    help_run = "Run the EPASWMM model"
    parser_run = subparsers.add_parser("run", help=help_run)
    parser_run.set_defaults(func=run_model)

    # create the parser for the "post" command
    help_pos = "Set the EPASWMM output files/states for Delft-FEWS"
    parser_post = subparsers.add_parser("post", help=help_pos)
    parser_post.set_defaults(func=post_adapter)

    # set args for testing
    args = parser.parse_args()
    args.func()
