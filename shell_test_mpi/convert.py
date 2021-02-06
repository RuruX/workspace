import meshio

#mesh = meshio.read(
#    "conventional-co3.STL",  # string, os.PathLike, or a buffer/open file
#    file_format="stl"  # optional if filename is a path; inferred from extension
#)
#
#msh = meshio.read("conventional-co3.STL")
#meshio.write("mesh2.xdmf",msh)

mesh = meshio.read(
    "Airbus-A320-84000.stl",  # string, os.PathLike, or a buffer/open file
    file_format="stl"  # optional if filename is a path; inferred from extension
)

msh = meshio.read("Airbus-A320-84000.stl")
meshio.write("Airbus-A320-84000.xdmf",msh)
