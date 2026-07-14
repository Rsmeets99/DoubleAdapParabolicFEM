#pragma once

#include <array>
#include <cstddef>

namespace finite_element::nodes
{
    template<int P>
    struct EquidistantNodes
    {
        static_assert(P >= 1, "EquidistantNodes requires P >= 1.");

        static constexpr int N = P + 1;
        static constexpr std::size_t n_nodes = static_cast<std::size_t>(N);

        using Point = std::array<double, 1>;
        using FaceList = std::array<int, 2>;

        static constexpr std::size_t index(int i) noexcept
        {
            return static_cast<std::size_t>(i);
        }

        struct NodeMeta
        {
            FaceList faces{-1, -1};
            int num_faces = 0;
            int vertex = -1; // 0 = left endpoint, 1 = right endpoint, -1 = interior
        };

        struct GeneratedData
        {
            std::array<Point, n_nodes> points{};
        };

        static constexpr GeneratedData generate_data()
        {
            GeneratedData data{};

            for (int i = 0; i <= P; ++i)
            {
                data.points[index(i)] = {
                    static_cast<double>(i) / static_cast<double>(P)
                };
            }

            return data;
        }

        static constexpr GeneratedData generated = generate_data();
        static constexpr auto points = generated.points;

        static constexpr NodeMeta make_node_meta(int i)
        {
            NodeMeta m{};

            if (i == 0)
            {
                m.faces[index(m.num_faces++)] = 0;
                m.vertex = 0;
            }

            if (i == P)
            {
                m.faces[index(m.num_faces++)] = 1;
                m.vertex = 1;
            }

            return m;
        }

        static constexpr std::array<NodeMeta, n_nodes> generate_node_meta()
        {
            std::array<NodeMeta, n_nodes> meta{};
            for (int i = 0; i < N; ++i)
                meta[index(i)] = make_node_meta(i);
            return meta;
        }

        static constexpr auto node_meta = generate_node_meta();

        static constexpr std::array<int, 2> generate_vertex_nodes()
        {
            return {0, P};
        }

        static constexpr auto vertex_nodes = generate_vertex_nodes();

        // Endpoint nodes in natural left-to-right order.
        static constexpr std::array<std::array<int, 1>, 2> generate_face_nodes()
        {
            return {{{0}, {P}}};
        }

        static constexpr auto face_nodes = generate_face_nodes();

        static constexpr bool node_on_face(int node_id, int face)
        {
            if (face == 0) return node_id == 0;
            if (face == 1) return node_id == P;
            return false;
        }

        static constexpr bool node_is_vertex(int node_id)
        {
            return node_meta[index(node_id)].vertex >= 0;
        }

        static constexpr int vertex_of_node(int node_id)
        {
            return node_meta[index(node_id)].vertex;
        }

        static constexpr bool node_on_vertex(int node_id, int vertex)
        {
            return vertex_of_node(node_id) == vertex;
        }
    };
}
