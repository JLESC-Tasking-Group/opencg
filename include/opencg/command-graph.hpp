/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
**
** This software is a computer program whose purpose is to execute
** blas subroutines on multi-GPUs system.
**
** This software is governed by the CeCILL-C license under French law and
** abiding by the rules of distribution of free software.  You can  use,
** modify and/ or redistribute the software under the terms of the CeCILL-C
** license as circulated by CEA, CNRS and INRIA at the following URL
** "http://www.cecill.info".

** As a counterpart to the access to the source code and  rights to copy,
** modify and redistribute granted by the license, users are provided only
** with a limited warranty  and the software's author,  the holder of the
** economic rights,  and the successive licensors  have only  limited
** liability.

** In this respect, the user's attention is drawn to the risks associated
** with loading,  using,  modifying and/or developing or reproducing the
** software by the user in light of its specific status of free software,
** that may mean  that it is complicated to manipulate,  and  that  also
** therefore means  that it is reserved for developers  and  experienced
** professionals having in-depth computer knowledge. Users are therefore
** encouraged to load and test the software's suitability as regards their
** requirements in conditions enabling the security of their systems and/or
** data to be ensured and,  more generally, to use and operate it in the
** same conditions as regards security.

** The fact that you are presently reading this means that you have had
** knowledge of the CeCILL-C license and that you accept its terms.
**/

#ifndef __OPENCG_COMMAND_GRAPH_HPP__
# define __OPENCG_COMMAND_GRAPH_HPP__

# include <opencg/bitset2d.hpp>
# include <opencg/command-graph-pass.hpp>
# include <opencg/command-type.hpp>
# include <opencg/device-type.hpp>
# include <opencg/namespace.hpp>

# include <algorithm>
# include <functional>
# include <list>
# include <queue>
# include <stack>

# include <assert.h>
# include <stdio.h>
# include <stdint.h>

# ifndef OCG_COMMAND_GRAPH_DEBUG
#  define OCG_COMMAND_GRAPH_DEBUG 1
# endif

OCG_NAMESPACE_BEGIN

/* struct command_t exists */
struct command_t;
struct command_graph_node_t;
struct command_graph_t;

/* Integer type to use for indexing command graph nodes */
typedef size_t command_graph_node_index_t;

/* Integer type to use for walk ids */
typedef int8_t command_graph_walk_id_t;

enum command_graph_walk_direction_t
{
    COMMAND_GRAPH_WALK_DIRECTION_FORWARD,
    COMMAND_GRAPH_WALK_DIRECTION_BACKWARD
};

enum command_graph_walk_search_t
{
    COMMAND_GRAPH_WALK_SEARCH_DFS,
    COMMAND_GRAPH_WALK_SEARCH_BFS
};

/* Optimize contractions with hints */
enum command_graph_contraction_hint_t
{
    /* u,v, are false-twins */
    COMMAND_GRAPH_CONTRACTION_HINT_FALSE_TWINS  = (1 << 0),

    /* u,v are a sequence u -> v, with
     *  - 'u' having a single successor   (v)
     *  - 'v' having a single predecessor (u) */
    COMMAND_GRAPH_CONTRACTION_HINT_U_V_SEQUENCE = (1 << 1),
    COMMAND_GRAPH_CONTRACTION_HINT_V_U_SEQUENCE = (1 << 2),

    /* Contract {u,v} in place to that u := u (+) v, return u.
     * Else, return a new node w != u */
    COMMAND_GRAPH_CONTRACTION_HINT_INPLACE      = (1 << 3)
};

inline constexpr command_graph_contraction_hint_t
operator|(command_graph_contraction_hint_t a, command_graph_contraction_hint_t b)
{
    return static_cast<command_graph_contraction_hint_t>(
        static_cast<int>(a) | static_cast<int>(b)
    );
}

/* a node */
struct command_graph_node_t
{
    /* the associated command */
    command_t * command;

    /* the device that must execute the command */
    device_unique_id_t device_unique_id;

    /* the list of predecessor commands */
    std::list<command_graph_node_t *> predecessors;

    /* the list of successor commands */
    std::list<command_graph_node_t *> successors;

    /* index in the last iterator returned */
    command_graph_node_index_t iterator_index;

    /* dfs id */
    command_graph_walk_id_t walk_id;

    /* constructor/destructor */
    command_graph_node_t(
        command_t * command,
        const device_unique_id_t device_unique_id
    ) :
        command(command),
        device_unique_id(device_unique_id),
        predecessors(),
        successors(),
        walk_id(0)
    {}

    /* insert this node to precede the passed node */
    inline void
    precedes(command_graph_node_t * succ)
    {
        this->successors.push_back(succ);
        succ->predecessors.push_back(this);
    }

    /* insert this node to succeed the passed node */
    inline void
    succeed(command_graph_node_t * pred)
    {
        pred->precedes(this);
    }

    inline void
    foreach_predecessor(std::function<void(command_graph_node_t * pred)> f)
    {
        for (command_graph_node_t * node : this->predecessors)
        {
            assert(node != this); // DAG
            f(node);
        }
    }

    inline void
    foreach_successor(std::function<void(command_graph_node_t * succ)> f)
    {
        for (command_graph_node_t * node : this->successors)
        {
            assert(node != this); // DAG
            f(node);
        }
    }

    template <command_graph_walk_direction_t direction = COMMAND_GRAPH_WALK_DIRECTION_FORWARD,
              command_graph_walk_search_t    search    = COMMAND_GRAPH_WALK_SEARCH_DFS>
    inline void
    walk(std::function<void(command_graph_node_t * node)> f)
    {
        /* lambda to enqueue unvisited neighbors into the frontier,
         * traversing successors (FORWARD) or predecessors (BACKWARD) */
        auto visit_neighbors = [&] (auto & frontier, command_graph_node_t * node)
        {
            auto enqueue = [&] (command_graph_node_t * neighbor)
            {
                /* skip already-visited nodes */
                if (neighbor->walk_id == this->walk_id)
                    return ;
                neighbor->walk_id = this->walk_id;
                frontier.push(neighbor);
            };

            if constexpr (direction == COMMAND_GRAPH_WALK_DIRECTION_FORWARD)
                node->foreach_successor(enqueue);
            else
            {
                static_assert(direction == COMMAND_GRAPH_WALK_DIRECTION_BACKWARD);
                node->foreach_predecessor(enqueue);
            }
        };

        if constexpr (search == COMMAND_GRAPH_WALK_SEARCH_DFS)
        {
            /* DFS: last-in, first-out via std::stack */
            std::stack<command_graph_node_t *> frontier;
            frontier.push(this);

            while (!frontier.empty())
            {
                command_graph_node_t * node = frontier.top();
                frontier.pop();

                assert(node->walk_id == this->walk_id);
                f(node);

                visit_neighbors(frontier, node);
            }
        }
        else
        {
            static_assert(search == COMMAND_GRAPH_WALK_SEARCH_BFS);

            /* BFS: first-in, first-out via std::queue */
            std::queue<command_graph_node_t *> frontier;
            frontier.push(this);

            while (!frontier.empty())
            {
                command_graph_node_t * node = frontier.front();
                frontier.pop();

                assert(node->walk_id == this->walk_id);
                f(node);

                visit_neighbors(frontier, node);
            }
        }
    }
};

/* Allocator types */
typedef command_t * (*command_allocator_t)(command_graph_t * cg, command_type_t type);
typedef command_graph_node_t * (*command_graph_node_allocator_t)(command_graph_t * cg, command_t * command, const device_unique_id_t device_unique_id);
typedef command_graph_t * (*command_graph_allocator_t)(command_graph_t * cg);

/**
 *  Represent a graph of commands to execute.
 */
struct command_graph_t
{
    /* entry and exit nodes */
    command_graph_node_t * entry;
    command_graph_node_t * exit;

    /* dfs id */
    command_graph_walk_id_t walk_id;

    /** Methods to allocate command, nodes and graphs */
    command_allocator_t command_new;

    /* Create a new command graph node */
    command_graph_node_allocator_t command_graph_node_new;

    /* Create a new command graph */
    command_graph_allocator_t command_graph_new;

    /* allocate initial entry/exit nodes */
    inline void
    init(
        command_allocator_t command_new,
        command_graph_node_allocator_t command_graph_node_new,
        command_graph_allocator_t command_graph_new
    ) {
        this->command_new = command_new;
        this->command_graph_node_new = command_graph_node_new;
        this->command_graph_new = command_graph_new;

        this->entry = this->command_graph_node_new(this, NULL, OCG_UNSPECIFIED_DEVICE_UNIQUE_ID);
        this->exit  = this->command_graph_node_new(this, NULL, OCG_UNSPECIFIED_DEVICE_UNIQUE_ID);
        assert(this->entry);
        assert(this->exit);
        this->entry->precedes(this->exit);
        this->walk_id = 0;
    }

    /* coherence checks */
    void coherence_checks(void);

    # define DEF(ENUM, FUNC, NAME) void FUNC(void);
    OCG_FORALL_COMMAND_GRAPH_PASS(DEF);
    # undef DEF

    /* Run an passimization pass */
    inline void
    optimize(command_graph_pass_t pass)
    {
        # if OCG_COMMAND_GRAPH_DEBUG
        char fname[128];
        const char * name = command_graph_pass_to_str(pass);
        {
            snprintf(fname, sizeof(fname), "cg-pre-%s.dot", name);
            FILE * f = fopen(fname, "w");
            this->dump(f);
            fclose(f);
        }
        # endif

        switch (pass)
        {
            # define DEF(ENUM, FUNC, NAME) case(ENUM): { this->FUNC(); break; }
            OCG_FORALL_COMMAND_GRAPH_PASS(DEF);
            # undef DEF
            default:
            {
                fprintf(stderr, "Pass not existing\n");
                abort();
            }
        }

        # if OCG_COMMAND_GRAPH_DEBUG
        {
            this->coherence_checks();
            snprintf(fname, sizeof(fname), "cg-post-%s.dot", name);
            FILE * f = fopen(fname, "w");
            this->dump(f);
            fclose(f);
        }
        # endif
    }

    /* get entry node */
    inline command_graph_node_t *
    node_get_entry(void)
    {
        return this->entry;
    }

    /* get exit node */
    inline command_graph_node_t *
    node_get_exit(void)
    {
        return this->exit;
    }

    /* set entry node */
    inline void
    node_set_entry(command_graph_node_t * node)
    {
        assert(node->predecessors.size() == 0);
        this->entry = node;
    }

    /* set exit node */
    inline void
    node_set_exit(command_graph_node_t * node)
    {
        assert(node->successors.size() == 0);
        this->exit = node;
    }

    /* return true if u and v are false twins (i.e., have the same
     * neighborhood, and are not connected) */
    inline bool
    are_false_twins(
        command_graph_node_t * u,
        command_graph_node_t * v
    ) {
        return (std::is_permutation(u->predecessors.begin(), u->predecessors.end(), v->predecessors.begin(), v->predecessors.end()) &&
                std::is_permutation(u->successors.begin(),   u->successors.end(),   v->successors.begin(),   v->successors.end()));
    }

    inline bool
    are_sequence(
        command_graph_node_t * u,
        command_graph_node_t * v
    ) {
        return (u->successors.size() == 1 && u->successors.front() == v) &&
            (v->predecessors.size() == 1 && v->predecessors.front() == u);
    }

    /* remove a node */
    inline void
    remove(command_graph_node_t * u)
    {
        /* remove every edges to u */
        for (command_graph_node_t * pred : u->predecessors)
            pred->successors.erase(std::find(pred->successors.begin(), pred->successors.end(), u));

        for (command_graph_node_t * succ : u->successors)
            succ->predecessors.erase(std::find(succ->predecessors.begin(), succ->predecessors.end(), u));

        /* connect u's predecessors with u's successors */
        for (command_graph_node_t * pred : u->predecessors)
        {
            for (command_graph_node_t * succ : u->successors)
            {
                /* only add if they are not already connected */
                auto it = std::find(pred->successors.begin(), pred->successors.end(), succ);
                if (it == pred->successors.end())
                    pred->precedes(succ);
            }
        }
    }

    /**
     *  Contract 2 nodes of the graph.
     *  If hints are set, it accelerates this routine.
     *
     *  You may provide these if the nodes {u,v} matches
     *      - COMMAND_GRAPH_CONTRACTION_HINT_FALSE_TWINS    u    v
     *      - COMMAND_GRAPH_CONTRACTION_HINT_U_V_SEQUENCE   u -> v
     *      - COMMAND_GRAPH_CONTRACTION_HINT_V_U_SEQUENCE   v -> u
     *
     *  You may provide
     *      - COMMAND_GRAPH_CONTRACTION_HINT_INPLACE
     *  to contract inplace of u.
     *
     *  If COMMAND_GRAPH_CONTRACTION_HINT_FALSE_TWINS is provided,
     *    u,v are contracted 'inplace' of u and u is returned.
     *    accessing v's precedessor/successor lists becomes UB
     *  Else
     *    a new node w is spawned as the contraction of u,v.
     *    accessing u's or v's predecessor/successor lists becomes UB.
     */
    template <command_graph_contraction_hint_t hints = 0>
    inline command_graph_node_t *
    contract(
        command_graph_node_t * u,   /* INOUT */
        command_graph_node_t * v    /* IN */
    ) {
        /* coherence tests */
        assert(u != v);

        /*************************************
         * Reconnect predecessors/successors *
         *************************************/

        /* if u,v are false-twins, the there is nothing to do */
        if constexpr (hints & COMMAND_GRAPH_CONTRACTION_HINT_FALSE_TWINS)
        {
            assert(this->are_false_twins(u, v));

            /* Remove u,v, from predecessors/successosrs lists, and reconnect to w */

            /* if in place, just remove 'v' from predecessors/successors lists, keep 'u', reutrn 'u' */
            if constexpr (hints & COMMAND_GRAPH_CONTRACTION_HINT_INPLACE)
            {
                /* predecessors */
                u->foreach_predecessor([&] (command_graph_node_t * pred) {
                    assert(std::count(pred->successors.begin(), pred->successors.end(), v) == 1);
                    pred->successors.erase(std::find(pred->successors.begin(), pred->successors.end(), v));
                });

                /* successors */
                u->foreach_successor([&] (command_graph_node_t * succ) {
                    assert(std::count(succ->predecessors.begin(), succ->predecessors.end(), v) == 1);
                    succ->predecessors.erase(std::find(succ->predecessors.begin(), succ->predecessors.end(), v));
                });
                return u;
            }
            else
            {
                assert(this->command_graph_node_new);
                command_graph_node_t * w = this->command_graph_node_new(this, NULL, OCG_UNSPECIFIED_DEVICE_UNIQUE_ID);
                assert(w);

                /* predecessors */
                u->foreach_predecessor([&] (command_graph_node_t * pred) {
                    assert(std::count(pred->successors.begin(), pred->successors.end(), u) == 1);
                    assert(std::count(pred->successors.begin(), pred->successors.end(), v) == 1);
                    pred->successors.erase(std::find(pred->successors.begin(), pred->successors.end(), u)); // TODO: 2 loops here, could be optimized
                    pred->successors.erase(std::find(pred->successors.begin(), pred->successors.end(), v));
                    pred->precedes(w);
                });

                /* successors */
                u->foreach_successor([&] (command_graph_node_t * succ) {
                    assert(std::count(succ->predecessors.begin(), succ->predecessors.end(), u) == 1);
                    assert(std::count(succ->predecessors.begin(), succ->predecessors.end(), v) == 1);
                    succ->predecessors.erase(std::find(succ->predecessors.begin(), succ->predecessors.end(), u)); // TODO: 2 loops here, could be optimized
                    succ->predecessors.erase(std::find(succ->predecessors.begin(), succ->predecessors.end(), v));
                    w->precedes(succ);
                });

                return w;
            }
        }
        /* u -> v */
        else if constexpr (hints & COMMAND_GRAPH_CONTRACTION_HINT_U_V_SEQUENCE)
        {
            assert(this->are_sequence(u, v));
            if constexpr (hints & COMMAND_GRAPH_CONTRACTION_HINT_INPLACE)
            {
                /* v successors' predecessors must point to u, not v anymore */
                v->foreach_successor([&] (command_graph_node_t * succ) {
                    assert(std::count(succ->predecessors.begin(), succ->predecessors.end(), v) == 1);
                    assert(succ != u);
                    auto it = std::find(succ->predecessors.begin(), succ->predecessors.end(), v);
                    assert(it != succ->predecessors.end());
                    *it = u;
                });

                /* u successors becomes v successors */
                u->successors = std::move(v->successors);

                return u;
            }
            else
            {
                assert(this->command_graph_node_new);
                command_graph_node_t * w = this->command_graph_node_new(this, NULL, OCG_UNSPECIFIED_DEVICE_UNIQUE_ID);
                assert(w);

                /* predecessors */
                u->foreach_predecessor([&] (command_graph_node_t * pred) {
                    assert(std::count(pred->successors.begin(), pred->successors.end(), u) == 1);
                    pred->successors.erase(std::find(pred->successors.begin(), pred->successors.end(), u));
                    pred->precedes(w);
                });

                /* successors */
                v->foreach_successor([&] (command_graph_node_t * succ) {
                    assert(std::count(succ->predecessors.begin(), succ->predecessors.end(), v) == 1);
                    succ->predecessors.erase(std::find(succ->predecessors.begin(), succ->predecessors.end(), v));
                    w->precedes(succ);
                });

                return w;
            }
        }
        /* v -> u */
        else if constexpr (hints & COMMAND_GRAPH_CONTRACTION_HINT_V_U_SEQUENCE)
        {
            assert(this->are_sequence(v, u));
            if constexpr (hints & COMMAND_GRAPH_CONTRACTION_HINT_INPLACE)
            {
                /* v predecessors' successors must point to u, not v anymore */
                v->foreach_predecessor([&] (command_graph_node_t * pred) {
                    assert(std::count(pred->successors.begin(), pred->successors.end(), v) == 1);
                    assert(pred != u);
                    auto it = std::find(pred->successors.begin(), pred->successors.end(), v);
                    assert(it != pred->successors.end());
                    *it = u;
                });

                /* u successors becomes v successors */
                u->predecessors = std::move(v->predecessors);

                return u;
            }
            else
            {
                assert(this->command_graph_node_new);
                command_graph_node_t * w = this->command_graph_node_new(this, NULL, OCG_UNSPECIFIED_DEVICE_UNIQUE_ID);
                assert(w);

                /* predecessors */
                v->foreach_predecessor([&] (command_graph_node_t * pred) {
                    assert(std::count(pred->successors.begin(), pred->successors.end(), v) == 1);
                    pred->successors.erase(std::find(pred->successors.begin(), pred->successors.end(), v));
                    pred->precedes(w);
                });

                /* successors */
                u->foreach_successor([&] (command_graph_node_t * succ) {
                    assert(std::count(succ->predecessors.begin(), succ->predecessors.end(), u) == 1);
                    succ->predecessors.erase(std::find(succ->predecessors.begin(), succ->predecessors.end(), u));
                    w->precedes(succ);
                });

                return w;
            }
        }
        else // generic case
        {
            abort();
            # if 0
            /*********************************************************************************************/
            /* connect v's predecessors/successors to u, if they are not already predecessors/successors */
            /*********************************************************************************************/

            /* annotate neighbors wih 'u' to avoid setting redundant edges */
            this->foreach_neighbor(u, [&] (command_graph_node_t * neigh) {
                neigh->toto = u;
            });
            u->toto = u;

            /* predecessors */
            this->foreach_predecessor(v, [&] (command_graph_node_t * pred) {
                if (pred->toto == u)
                    continue ;

                /* disconnect from u (careful: we are iterating through v,
                 * therefore only pop from others' lists, not from v. moreover, we
                 * dont really care about these lists in 'v' from now on */

                /* a node should never appear more than once in these lists */
                assert(std::count(pred->successors.begin(), pred->successors.end(), v) <= 1);

                /* remove it if it appears */
                auto it = std::find(pred->successors.begin(), pred->successors.end(), v);
                if (it != pred->successors.end())
                    pred->successors.erase(it);

                /* connect to u */
                pred->precedes(u);
            });

            /* successors */
            this->foreach_successor(v, [&] (command_graph_node_t * succ) {
                if (succ->toto == u)
                    continue ;

                assert(std::count(succ->predecessors.begin(), succ->predecessors.end(), v) <= 1);
                auto it = std::find(succ->predecessors.begin(), succ->predecessors.end(), v);
                if (it != succ->predecessors.end())
                    succ->predecessors.erase(it);

                u->precedes(succ);
            });

            /* Remove edge (u -> v) or (v -> u) */
            # endif

            return u;
        }
    }

    template <command_graph_walk_direction_t direction = COMMAND_GRAPH_WALK_DIRECTION_FORWARD,
              command_graph_walk_search_t    search    = COMMAND_GRAPH_WALK_SEARCH_DFS>
    inline void
    walk(
        command_graph_node_t * node,
        std::function<void(command_graph_node_t * node)> f
    ) {
        node->walk_id = ++this->walk_id;
        node->walk<direction, search>(f);
    }

    template <command_graph_walk_direction_t direction = COMMAND_GRAPH_WALK_DIRECTION_FORWARD,
              command_graph_walk_search_t    search    = COMMAND_GRAPH_WALK_SEARCH_DFS>
    inline void
    walk(std::function<void(command_graph_node_t * node)> f)
    {
        command_graph_node_t * node;
        if constexpr (direction == COMMAND_GRAPH_WALK_DIRECTION_FORWARD)
        {
            node = this->node_get_entry();
        }
        else
        {
            static_assert(direction == COMMAND_GRAPH_WALK_DIRECTION_BACKWARD);
            node = this->node_get_exit();
        }
        return this->walk<direction, search>(node, f);
    }

    /* Dump the command graph */
    void dump(FILE * file);

    /**************
     * Iterators *
     *************/

    template<typename T>
    struct node_iterator_t
    {
        /* the node */
        command_graph_node_t * node;

        /* the node local storage */
        T data;

        node_iterator_t(command_graph_node_t * node) : node(node), data() {}
    };

    template<typename T,
             bool include_entry_exit = false,
             command_graph_walk_direction_t direction = COMMAND_GRAPH_WALK_DIRECTION_FORWARD,
             command_graph_walk_search_t    search    = COMMAND_GRAPH_WALK_SEARCH_DFS>
    inline std::vector<node_iterator_t<T>>
    create_node_iterators(const command_graph_node_index_t initial_capacity = 2048)
    {
        /* create a vector */
        std::vector<node_iterator_t<T>> vec;
        vec.reserve(initial_capacity);

        /* walk through the graph */
        this->walk<direction, search>(
            [&] (command_graph_node_t * node)
            {
                if constexpr (!include_entry_exit)
                    if (node == this->node_get_entry() || node == this->node_get_exit())
                        return ;
                node->iterator_index = vec.size();
                vec.push_back(node);
            }
        );
        return vec;
    }

};

OCG_NAMESPACE_END

#endif /* __OPENCG_COMMAND_GRAPH_HPP__ */
