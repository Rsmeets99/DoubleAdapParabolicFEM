#pragma once

#include <array>
#include <cstddef>

namespace finite_element::nodes
{
    template<int P>
    struct EquidistantTriangleNodes
    {
        static_assert(P >= 1, "EquidistantTriangleNodes requires P >= 1.");

        static constexpr int N = (P + 1) * (P + 2) / 2;
        static constexpr std::size_t n_nodes = static_cast<std::size_t>(N);
        static constexpr std::size_t n_face_nodes = static_cast<std::size_t>(P + 1);

        using Point    = std::array<double, 2>;
        using Bary     = std::array<int, 3>;   // {a,b,c}, a+b+c = P
        using FaceList = std::array<int, 2>;

        static constexpr std::size_t index(int i) noexcept
        {
            return static_cast<std::size_t>(i);
        }

        struct NodeMeta
        {
            FaceList spatial_faces{-1, -1};
            int num_spatial_faces = 0;
            int vertex = -1; // 0,1,2 or -1
        };

        struct GeneratedData
        {
            std::array<Point, n_nodes> points{};
            std::array<Bary, n_nodes> barycentric_tuples{};
        };

        static constexpr GeneratedData generate_data()
        {
            GeneratedData data{};

            int id = 0;
            for (int c = 0; c <= P; ++c)
            {
                for (int b = 0; b <= P - c; ++b)
                {
                    const int a = P - b - c;

                    const double x = static_cast<double>(b) / static_cast<double>(P);
                    const double y = static_cast<double>(c) / static_cast<double>(P);

                    data.points[index(id)] = {x, y};
                    data.barycentric_tuples[index(id)] = {a, b, c};
                    ++id;
                }
            }

            return data;
        }

        static constexpr GeneratedData generated = generate_data();
        static constexpr auto points = generated.points;
        static constexpr auto barycentric_tuples = generated.barycentric_tuples;

        static constexpr NodeMeta make_node_meta(const Bary& n)
        {
            NodeMeta m{};

            // face 0: c == 0  => y = 0
            if (n[2U] == 0)
                m.spatial_faces[index(m.num_spatial_faces++)] = 0;

            // face 1: a == 0  => x + y = 1
            if (n[0U] == 0)
                m.spatial_faces[index(m.num_spatial_faces++)] = 1;

            // face 2: b == 0  => x = 0
            if (n[1U] == 0)
                m.spatial_faces[index(m.num_spatial_faces++)] = 2;

            // vertices:
            // v0 = (0,0) -> {P,0,0}
            // v1 = (1,0) -> {0,P,0}
            // v2 = (0,1) -> {0,0,P}
            if (n[0U] == P && n[1U] == 0 && n[2U] == 0) m.vertex = 0;
            if (n[0U] == 0 && n[1U] == P && n[2U] == 0) m.vertex = 1;
            if (n[0U] == 0 && n[1U] == 0 && n[2U] == P) m.vertex = 2;

            return m;
        }

        static constexpr std::array<NodeMeta, n_nodes> generate_node_meta()
        {
            std::array<NodeMeta, n_nodes> meta{};
            for (int i = 0; i < N; ++i)
                meta[index(i)] = make_node_meta(barycentric_tuples[index(i)]);
            return meta;
        }

        static constexpr auto node_meta = generate_node_meta();

        static constexpr std::array<std::array<int, n_face_nodes>, 3> generate_face_nodes()
        {
            std::array<std::array<int, n_face_nodes>, 3> generated_face_nodes{};

            for (auto& face : generated_face_nodes)
                for (auto& x : face)
                    x = -1;

            for (int i = 0; i < N; ++i)
            {
                const Bary n = barycentric_tuples[index(i)];
                const int a = n[0U];
                const int b = n[1U];
                const int c = n[2U];

                // face 0: (0,0) -> (1,0), c == 0, order by b increasing
                if (c == 0)
                    generated_face_nodes[0U][index(b)] = i;

                // face 1: (1,0) -> (0,1), a == 0, order by c increasing
                if (a == 0)
                    generated_face_nodes[1U][index(c)] = i;

                // face 2: (0,1) -> (0,0), b == 0, order by c decreasing = a increasing
                if (b == 0)
                    generated_face_nodes[2U][index(a)] = i;
            }

            return generated_face_nodes;
        }

        static constexpr auto face_nodes = generate_face_nodes();

        static constexpr std::array<std::array<int, 3>, n_nodes> generate_face_ordinal()
        {
            std::array<std::array<int, 3>, n_nodes> ord{};

            for (auto& row : ord)
                row = {-1, -1, -1};

            for (int face = 0; face < 3; ++face)
            {
                for (int k = 0; k <= P; ++k)
                {
                    const int node = face_nodes[index(face)][index(k)];
                    ord[index(node)][index(face)] = k;
                }
            }

            return ord;
        }

        static constexpr auto face_ordinal = generate_face_ordinal();

        static constexpr std::array<int, 3> generate_vertex_nodes()
        {
            std::array<int, 3> verts{-1, -1, -1};

            for (int i = 0; i < N; ++i)
            {
                const int v = node_meta[index(i)].vertex;
                if (v >= 0)
                    verts[index(v)] = i;
            }

            return verts;
        }

        static constexpr auto vertex_nodes = generate_vertex_nodes();

        static constexpr bool node_on_face(int node_id, int face)
        {
            return face_ordinal[index(node_id)][index(face)] >= 0;
        }

        static constexpr bool node_is_vertex(int node_id)
        {
            return node_meta[index(node_id)].vertex >= 0;
        }

        static constexpr int vertex_of_node(int node_id)
        {
            return node_meta[index(node_id)].vertex;
        }
    };
    
}
