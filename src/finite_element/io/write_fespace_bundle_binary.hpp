#pragma once

#include <filesystem>
#include <string>

template<class SpaceType>
void write_fespace_bundle_binary(
    const SpaceType& space,
    const std::filesystem::path& output_dir,
    const std::string& mesh_filename = "mesh.bin",
    const std::string& dofs_filename = "dofs.bin")
{
    space.write_mesh_binary(output_dir, mesh_filename);
    space.write_dofs_binary(output_dir, dofs_filename);
}