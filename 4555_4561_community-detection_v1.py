import itertools
import pandas as pd
import numpy as np
import random
import networkx as nx
from networkx.algorithms.community import girvan_newman, modularity
from numpy import partition
import time
import matplotlib.pyplot as plt
from networkx.algorithms import community
import os


############################### FG COLOR DEFINITIONS ###############################
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'  # RECOVERS DEFAULT TEXT COLOR
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''


########################################################################################
############################## MY ROUTINES LIBRARY STARTS ##############################
########################################################################################

# SIMPLE ROUTINE TO CLEAR SCREEN BEFORE SHOWING A MENU...
def my_clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


# CREATE A LIST OF RANDOMLY CHOSEN COLORS...
def my_random_color_list_generator(REQUIRED_NUM_COLORS):
    my_color_list = ['red',
                     'green',
                     'cyan',
                     'brown',
                     'olive',
                     'orange',
                     'darkblue',
                     'purple',
                     'yellow',
                     'hotpink',
                     'teal',
                     'gold']

    my_used_colors_dict = {c: 0 for c in
                           my_color_list}  # DICTIONARY OF FLAGS FOR COLOR USAGE. Initially no color is used...
    constructed_color_list = []

    if REQUIRED_NUM_COLORS <= len(my_color_list):
        for i in range(REQUIRED_NUM_COLORS):
            constructed_color_list.append(my_color_list[i])

    else:  # REQUIRED_NUM_COLORS > len(my_color_list)
        constructed_color_list = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in
                                  range(REQUIRED_NUM_COLORS)]

    return (constructed_color_list)


# VISUALISE A GRAPH WITH COLORED NODES AND LINKS
def my_graph_plot_routine(G, fb_nodes_colors, fb_links_colors, fb_links_styles, graph_layout, node_positions):
    plt.figure(figsize=(10, 10))

    if len(node_positions) == 0:
        if graph_layout == 'circular':
            node_positions = nx.circular_layout(G)
        elif graph_layout == 'random':
            node_positions = nx.random_layout(G, seed=50)
        elif graph_layout == 'planar':
            node_positions = nx.planar_layout(G)
        elif graph_layout == 'shell':
            node_positions = nx.shell_layout(G)
        else:  # DEFAULT VALUE == spring
            node_positions = nx.spring_layout(G)

    nx.draw(G,
            with_labels=True,  # indicator variable for showing the nodes' ID-labels
            style=fb_links_styles,  # edge-list of link styles, or a single default style for all edges
            edge_color=fb_links_colors,  # edge-list of link colors, or a single default color for all edges
            pos=node_positions,  # node-indexed dictionary, with position-values of the nodes in the plane
            node_color=fb_nodes_colors,  # either a node-list of colors, or a single default color for all nodes
            node_size=100,  # node-circle radius
            alpha=0.9,  # fill-transparency
            width=0.5  # edge-width
            )
    plt.show()

    return (node_positions)


########################################################################################
# MENU 1 STARTS: creation of input graph ### 
########################################################################################
def my_menu_graph_construction(G, node_names_list, node_positions):
    my_clear_screen()

    breakWhileLoop = False

    while not breakWhileLoop:
        print(bcolors.OKGREEN
              + '''
========================================
(1.1) Create graph from fb-food data set (fb-pages-food.nodes and fb-pages-food.nodes)\t[format: L,<NUM_LINKS>]
(1.2) Create RANDOM Erdos-Renyi graph G(n,p).\t\t\t\t\t\t[format: R,<number of nodes>,<edge probability>]
(1.3) Print graph\t\t\t\t\t\t\t\t\t[format: P,<GRAPH LAYOUT in {spring,random,circular,shell }>]    
(1.4) Continue with detection of communities.\t\t\t\t\t\t[format: N]
(1.5) EXIT\t\t\t\t\t\t\t\t\t\t[format: E]
----------------------------------------
        ''' + bcolors.ENDC)

        my_option_list = str(input('\tProvide your (case-sensitive) option: ')).split(',')

        if my_option_list[0] == 'L':
            MAX_NUM_LINKS = 2102  # this is the maximum number of links in the fb-food-graph data set...

            if len(my_option_list) > 2:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Too many parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

            else:
                if len(my_option_list) == 1:
                    NUM_LINKS = MAX_NUM_LINKS
                else:  # ...len(my_option_list) == 2...
                    NUM_LINKS = int(my_option_list[1])

                if NUM_LINKS > MAX_NUM_LINKS or NUM_LINKS < 1:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(
                        bcolors.WARNING + "\tERROR Invalid number of links to read from data set. It should be in {1,2,...,2102}. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                else:
                    # LOAD GRAPH FROM DATA SET...
                    # G,node_names_list =
                    G = STUDENT_4555_4561_read_graph_from_csv(NUM_LINKS)
                    print("\tConstructing the FB-FOOD graph with n =", G.number_of_nodes(),
                          "vertices and m =", G.number_of_edges(), "edges (after removal of loops).")

        elif my_option_list[0] == 'R':

            if len(my_option_list) > 3:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

            else:  # ...len(my_option_list) <= 3...
                if len(my_option_list) == 1:
                    NUM_NODES = 100  # DEFAULT NUMBER OF NODES FOR THE RANDOM GRAPH...
                    ER_EDGE_PROBABILITY = 2 / NUM_NODES  # DEFAULT VALUE FOR ER_EDGE_PROBABILITY...

                elif len(my_option_list) == 2:
                    NUM_NODES = int(my_option_list[1])
                    ER_EDGE_PROBABILITY = 2 / max(1, NUM_NODES)  # AVOID DIVISION WITH ZERO...

                else:  # ...NUM_NODES == 3...
                    NUM_NODES = int(my_option_list[1])
                    ER_EDGE_PROBABILITY = float(my_option_list[2])

                if ER_EDGE_PROBABILITY < 0 or ER_EDGE_PROBABILITY > 1 or NUM_NODES < 2:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(
                        bcolors.WARNING + "\tERROR MESSAGE: Invalid probability mass or number of nodes of G(n,p). Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    G = nx.erdos_renyi_graph(NUM_NODES, ER_EDGE_PROBABILITY)
                    print(bcolors.ENDC + "\tConstructing random Erdos-Renyi graph with n =", G.number_of_nodes(),
                          "vertices and edge probability p =", ER_EDGE_PROBABILITY,
                          "which resulted in m =", G.number_of_edges(), "edges.")

                    node_names_list = [x for x in range(NUM_NODES)]

        elif my_option_list[0] == 'P':  # PLOT G...
            print("Printing graph G with", G.number_of_nodes(), "vertices,", G.number_of_edges(), "edges and",
                  nx.number_connected_components(G), "connected components.")

            if len(my_option_list) > 3:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

            else:
                if len(my_option_list) <= 1:
                    graph_layout = 'spring'  # ...DEFAULT graph_layout value...
                    reset_node_positions = 'Y'  # ...DEFAULT decision: erase node_positions...

                elif len(my_option_list) == 2:
                    graph_layout = str(my_option_list[1])
                    reset_node_positions = 'Y'  # ...DEFAULT decision: erase node_positions...

                else:  # ...len(my_option_list) == 3...
                    graph_layout = str(my_option_list[1])
                    reset_node_positions = str(my_option_list[2])

                if graph_layout not in ['spring', 'random', 'circular', 'shell']:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(
                        bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice for graph layout. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                elif reset_node_positions not in ['Y', 'y', 'N', 'n']:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(
                        bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible decision for resetting node positions. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if reset_node_positions in ['y', 'Y']:
                        node_positions = []  # ...ERASE previous node positions...

                    node_positions = my_graph_plot_routine(G, 'grey', 'blue', 'solid', graph_layout, node_positions)

        elif my_option_list[0] == 'N':
            NUM_NODES = G.number_of_nodes()
            if NUM_NODES == 0:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(
                    bcolors.WARNING + "\tERROR MESSAGE: You have not yet constructed a graph to work with. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            else:
                my_clear_screen()
                breakWhileLoop = True

        elif my_option_list[0] == 'E':
            quit()

        else:
            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible input was provided. Try again..." + bcolors.ENDC)
            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

    return (G, node_names_list, node_positions)


########################################################################################
# MENU 2: detect communities in the constructed graph 
########################################################################################
def my_menu_community_detection(G, node_names_list, node_positions, hierarchy_of_community_tuples):
    breakWhileLoop = False

    while not breakWhileLoop:
        print(bcolors.OKGREEN
              + '''
========================================
(2.1) Add random edges from each node\t\t\t[format: RE,<NUM_RANDOM_EDGES_PER_NODE>,<EDGE_ADDITION_PROBABILITY in [0,1]>]
(2.2) Add hamilton cycle (if graph is not connected)\t[format: H]
(2.3) Print graph\t\t\t\t\t[format: P,<GRAPH LAYOUT in { spring, random, circular, shell }>,<ERASE NODE POSITIONS in {Y,N}>]
(2.4) Compute communities with GIRVAN-NEWMAN\t\t[format: C,<ALG CHOICE in { O(wn),N(etworkx) }>,<GRAPH LAYOUT in {spring,random,circular,shell }>]
(2.5) Compute a binary hierarchy of communities\t\t[format: D,<NUM_DIVISIONS>,<GRAPH LAYOUT in {spring,random,circular,shell }>]
(2.6) Compute modularity-values for all community partitions\t[format: M]
(2.7) Visualize the communities of the graph\t\t[format: V,<GRAPH LAYOUT in {spring,random,circular,shell}>]
(2.8) EXIT\t\t\t\t\t\t[format: E]
----------------------------------------
            ''' + bcolors.ENDC)

        my_option_list = str(input('\tProvide your (case-sensitive) option: ')).split(',')

        if my_option_list[0] == 'RE':  # 2.1: ADD RANDOM EDGES TO NODES...

            if len(my_option_list) > 3:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(
                    bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. [format: D,<NUM_RANDOM_EDGES>,<EDGE_ADDITION_PROBABILITY>]. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

            else:
                if len(my_option_list) == 1:
                    NUM_RANDOM_EDGES = 1  # DEFAULT NUMBER OF RANDOM EDGES TO ADD (per node)
                    EDGE_ADDITION_PROBABILITY = 0.25  # DEFAULT PROBABILITY FOR ADDING EACH RANDOM EDGE (independently of other edges) FROM EAC NODE (independently from other nodes)...

                elif len(my_option_list) == 2:
                    NUM_RANDOM_EDGES = int(my_option_list[1])
                    EDGE_ADDITION_PROBABILITY = 0.25  # DEFAULT PROBABILITY FOR ADDING EACH RANDOM EDGE (independently of other edges) FROM EAC NODE (independently from other nodes)...

                else:
                    NUM_RANDOM_EDGES = int(my_option_list[1])
                    EDGE_ADDITION_PROBABILITY = float(my_option_list[2])

                # CHECK APPROPIATENESS OF INPUT AND RUN THE ROUTINE...
                if NUM_RANDOM_EDGES - 1 not in range(5):
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(
                        bcolors.WARNING + "\tERROR MESSAGE: Too many random edges requested. Should be from {1,2,...,5}. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                elif EDGE_ADDITION_PROBABILITY < 0 or EDGE_ADDITION_PROBABILITY > 1:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(
                        bcolors.WARNING + "\tERROR MESSAGE: Not appropriate value was given for EDGE_ADDITION PROBABILITY. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    G = STUDENT_4555_4561_add_random_edges_to_graph(G, node_names_list, NUM_RANDOM_EDGES,
                                                                    EDGE_ADDITION_PROBABILITY)

        elif my_option_list[0] == 'H':  # 2.2: ADD HAMILTON CYCLE...

            G = STUDENT_4555_4561_add_hamilton_cycle_to_graph(G, node_names_list)


        elif my_option_list[0] == 'P':  # 2.3: PLOT G...
            print("Printing graph G with", G.number_of_nodes(), "vertices,", G.number_of_edges(), "edges and",
                  nx.number_connected_components(G), "connected components.")

            if len(my_option_list) > 2:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

            else:
                if len(my_option_list) <= 1:
                    graph_layout = 'spring'  # ...DEFAULT graph_layout value...

                else:  # ...len(my_option_list) == 2...
                    graph_layout = str(my_option_list[1])

                if graph_layout not in ['spring', 'random', 'circular', 'shell']:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(
                        bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice for graph layout. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if len(my_option_list) == 2:
                        node_positions = []  # ...ERASE previous node positions...

                    node_positions = my_graph_plot_routine(G, 'grey', 'blue', 'solid', graph_layout, node_positions)

        elif my_option_list[0] == 'C':  # 2.4: COMPUTE ONE-SHOT GN-COMMUNITIES
            NUM_OPTIONS = len(my_option_list)

            if NUM_OPTIONS > 3:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

            else:
                if NUM_OPTIONS == 1:
                    alg_choice = 'N'  # DEFAULT COMM-DETECTION ALGORITHM == NX_GN
                    graph_layout = 'spring'  # DEFAULT graph layout == spring

                elif NUM_OPTIONS == 2:
                    alg_choice = str(my_option_list[1])
                    graph_layout = 'spring'  # DEFAULT graph layout == spring

                else:  # ...NUM_OPTIONS == 3...
                    alg_choice = str(my_option_list[1])
                    graph_layout = str(my_option_list[2])

                # CHECKING CORRECTNESS OF GIVWEN PARAMETERS...
                if alg_choice == 'N' and graph_layout in ['spring', 'circular', 'random', 'shell']:
                    STUDENT_4555_4561_use_nx_girvan_newman_for_communities(G, graph_layout, node_positions)

                elif alg_choice == 'O' and graph_layout in ['spring', 'circular', 'random', 'shell']:
                    STUDENT_4555_4561_one_shot_girvan_newman_for_communities(G, graph_layout, node_positions)

                else:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(
                        bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible parameters for executing the GN-algorithm. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

        elif my_option_list[0] == 'D':  # 2.5: COMUTE A BINARY HIERARCHY OF COMMUNITY PARRTITIONS
            NUM_OPTIONS = len(my_option_list)
            NUM_NODES = G.number_of_nodes()
            NUM_COMPONENTS = nx.number_connected_components(G)
            MAX_NUM_DIVISIONS = min(8 * NUM_COMPONENTS, np.floor(NUM_NODES / 4))

            if NUM_OPTIONS > 3:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

            else:
                if NUM_OPTIONS == 1:
                    number_of_divisions = 2 * NUM_COMPONENTS  # DEFAULT number of communities to look for
                    graph_layout = 'spring'  # DEFAULT graph layout == spring

                elif NUM_OPTIONS == 2:
                    number_of_divisions = int(my_option_list[1])
                    graph_layout = 'spring'  # DEFAULT graph layout == spring

                else:  # ...NUM_OPTIONS == 3...
                    number_of_divisions = int(my_option_list[1])
                    graph_layout = str(my_option_list[2])

                # CHECKING SYNTAX OF GIVEN PARAMETERS...
                if number_of_divisions < NUM_COMPONENTS or number_of_divisions > MAX_NUM_DIVISIONS:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: The graph has already", NUM_COMPONENTS,
                          "connected components." + bcolors.ENDC)
                    print(bcolors.WARNING + "\tProvide a number of divisions in { ", NUM_COMPONENTS, ",",
                          MAX_NUM_DIVISIONS, "}. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                elif graph_layout not in ['spring', 'random', 'circular', 'shell']:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(
                        bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice of a graph layout. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    STUDENT_4555_4561_divisive_community_detection(G, number_of_divisions, graph_layout, node_positions)

        elif my_option_list[
            0] == 'M':  # 2.6: DETERMINE PARTITION OF MIN-MODULARITY, FOR A GIVEN BINARY HIERARCHY OF COMMUNITY PARTITIONS
            STUDENT_4555_4561_determine_opt_community_structure(G, hierarchy_of_community_tuples)


        elif my_option_list[0] == 'V':  # 2.7: VISUALIZE COMMUNITIES WITHIN GRAPH

            NUM_OPTIONS = len(my_option_list)

            if NUM_OPTIONS > 2:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

            else:

                if NUM_OPTIONS == 1:
                    graph_layout = 'spring'  # DEFAULT graph layout == spring

                else:  # ...NUM_OPTIONS == 2...
                    graph_layout = str(my_option_list[1])

                if graph_layout not in ['spring', 'random', 'circular', 'shell']:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(
                        bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice of a graph layout. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    STUDENT_4555_4561_visualize_communities(G, community_tuples, graph_layout, node_positions)

        elif my_option_list[0] == 'E':
            # EXIT the program execution...
            quit()

        else:
            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible input was provided. Try again..." + bcolors.ENDC)
            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
    ### MENU 2 ENDS: detect communities in the constructed graph ### 


########################################################################################
############################### MY ROUTINES LIBRARY ENDS ############################### 
########################################################################################

########################################################################################
########################## STUDENT_4555_4561 ROUTINES LIBRARY STARTS ##########################
# FILL IN THE REQUIRED ROUTINES FROM THAT POINT ON...
########################################################################################

########################################################################################
global loopless_links_list, G, graph_nodes_list
loopless_links_list = []
graph_nodes_list = []


def STUDENT_4555_4561_read_graph_from_csv(NUM_LINKS):
    global loopless_links_list, G, node_names_list, graph_nodes_list
    # Load to dataframe NUM_LINK edges
    fb_links_df = pd.read_csv("fb-pages-food.edges", nrows=NUM_LINKS)
    print("\n FB LINKS DATAFRAME OF ", NUM_LINKS, " EDGES \n", fb_links_df)

    # Dataframe to list of tuples
    fb_links_tuples = fb_links_df.to_records(index=False)
    fb_links_tuples_list = list(fb_links_tuples)
    # print(fb_links_tuples_list)

    # Remove loop-edges
    # loopless_links_list = []
    for i in range(len(fb_links_tuples_list)):
        tup = fb_links_tuples_list[i]
        if tup[0] == tup[1]:
            print("\n Removed loop-edge: ", fb_links_tuples_list[i])
        else:
            loopless_links_list.append(tup)

    # print(loopless_links_list)

    # Create fb_links_loopless_df dataframe
    fb_links_loopless_df = pd.DataFrame.from_records(loopless_links_list, columns=['node_1', 'node_2'])
    print("\n FB LINKS LOOPLESS DATAFRAME \n", fb_links_loopless_df)

    """
    nodes_1 = list(fb_links_loopless_df['node_1'])
    nodes_2 = list(fb_links_loopless_df['node_2'])
    node_names_list = nodes_1 + nodes_2
    node_names_list = list(set(node_names_list))
    graph_nodes_list = node_names_list
    print("\n Node names list: ", node_names_list)
    """

    # Print the graph of fb_links_loopless_df
    G = nx.from_pandas_edgelist(fb_links_loopless_df, "node_1", "node_2", create_using=nx.Graph())
    my_graph_plot_routine(G, 'r', 'b', 'solid', 'circular', '')

    # List of node names
    node_names_list = list(G.nodes)
    graph_nodes_list = node_names_list
    print("\n Node names list: ", node_names_list)


    """
    print(bcolors.ENDC + "\t" + '''
        ########################################################################################
        # CREATE GRAPH FROM EDGE_CSV DATA 
        # ...(if needed) load all details for the nodes to the fb_nodes DATAFRAME
        # ...create DATAFRAME edges_df with the first MAX_NUM_LINKS, from the edges dataset
        # ...edges_df has one row per edge, and two columns, named 'node_1' and 'node_2
        # ...CLEANUP: remove each link (from the loaded ones) which is a LOOP
        # ...create node_names_list of NODE IDs that appear as terminal points of the loaded edges
        # ...create graph from the edges_df dataframe
        # ...return the constructed graph
        ########################################################################################
''')

    print(bcolors.ENDC + "\tUSEFUL FUNCTIONS:")

    print(  bcolors.ENDC + "\t\t The routine " 
            + bcolors.OKCYAN + "nx.from_pandas_edgelist(...) "
            + bcolors.ENDC 
            + '''creates the graph from a dataframe representing its edges,
                 one per edge, in two columns representing the tail-node 
                 (node_1) and the head-node (node_2) of the edge.\n''')
    """

    return G


######################################################################################################################
# ...(a) STUDENT_4555_4561 IMPLEMENTATION OF ONE-SHOT GN-ALGORITHM...
######################################################################################################################

def edge_to_remove(graph):
    G_dict = nx.edge_betweenness_centrality(graph)
    edge = ()

    # extract the edge with highest edge betweenness centrality score
    for key, value in sorted(G_dict.items(), key=lambda item: item[1], reverse=True):
        edge = key
        break

    return edge


def STUDENT_4555_4561_one_shot_girvan_newman_for_communities(G, graph_layout, node_positions):
    print(bcolors.ENDC
          + "\tCalling routine "
          + bcolors.HEADER + "STUDENT_4555_4561_one_shot_girvan_newman_for_communities(G,graph_layout,node_positions)\n"
          + bcolors.ENDC)

    start_time = time.time()

    communities_list = []

    connected_comps = nx.connected_components(G)
    connected_comps_nums = nx.number_connected_components(G)

    while connected_comps_nums == 1:
        G.remove_edge(edge_to_remove(G)[0], edge_to_remove(G)[1])
        connected_comps = nx.connected_components(G)
        connected_comps_nums = nx.number_connected_components(G)

    print("Number of connected components: ", connected_comps_nums)

    for i in connected_comps:
        communities_list.append(tuple(i))

    print("Communities List: ", communities_list)
    print("LC1: ", communities_list[0])
    print("LC2: ", communities_list[1])

    """
    print(bcolors.ENDC + "\t" + '''
        ######################################################################################################################
        # PROVIDE YOUR OWN ROUTINE WHICH CREATES K+1 COMMUNITIES FOR A GRAPH WITH K CONNECTED COMPONENTS, AS FOLLOWS:
        # 
        # INPUT: 
        #   G = the constructed graph to work with. 
        #   graph_layout = a string determining how to position the graph nodes in the plane.
        #   node_positions = a list of previously computed node positions in the plane (for next call of my_graph_plot_routine)
        # OUTPUT: 
        #    community_tuples = a list of K+1 tuples of node_IDs, each tuple determining a different community
        # PSEUDOCODE:    
        # ...THE K CONNECTED COMPONENTS OF G ARE COMPUTED. 
        # ...THE LIST community_tuples IS INITIALIZED, WITH ONE NODES-TUPLE FOR EACH CONNECTED COMPONENT DEFINING A DIFFERENT COMMUNITY.
        # ...GCC = THE SUBGRAPH OF G INDUCED BY THE NODES OF THE LARGEST COMMUNITY, LC.
        # ...SPLIT LC IN TWO SUBCOMMUNITIES, BY REPEATEDLY REMOVING MAX_BETWEENNESS EDGES FROM GCC, UNTIL ITS DISCONNECTION.
        # ...THE NODE-LISTS OF THE TWO COMPONENTS OF GCC (AFTER THE EDGE REMOVALS) ARE THE NEW SUBCOMMUNITIES THAT SUBSTITUTE LC 
        # IN THE LIST community_tuples, AS SUGGESTED BY THE GIRVAN-NEWMAN ALGORITHM...
        ######################################################################################################################
    ''')
    """

    """
    print(bcolors.ENDC + "\tUSEFUL FUNCTIONS:")

    print(
        bcolors.ENDC + "\t\t" + bcolors.OKCYAN + "sorted(nx.connected_components(G), key=len, reverse=True) " + bcolors.ENDC
        + "initiates the community_tuples with the node-sets of the connected components of the graph G, sorted by their size")

    print(bcolors.ENDC + "\t\t" + bcolors.OKCYAN + "G.subgraph(X).copy() "
          + bcolors.ENDC + "creates the subgraph of G induced by a subset (even in list format) X of nodes.")

    print(bcolors.ENDC + "\t\t" + bcolors.OKCYAN + "edge_betweenness_centrality(...) "
          + bcolors.ENDC + "of networkx.algorithms.centrality computes edge-betweenness values.")

    print(bcolors.ENDC + "\t\t" + bcolors.OKCYAN + "is_connected(G) "
          + bcolors.ENDC + "of networkx checks connectedness of G.\n")
    """

    end_time = time.time()

    print(bcolors.ENDC + "\t===================================================")
    print(bcolors.ENDC + "\tYOUR OWN computation of ONE-SHOT Girvan-Newman clustering for a graph with",
          G.number_of_nodes(), "nodes and", G.number_of_edges(), "links. "
          + "Computation time =", end_time - start_time, "\n")

    return communities_list


######################################################################################################################
# ...(b) USE NETWORKX IMPLEMENTATION OF ONE-SHOT GN-ALGORITHM...
######################################################################################################################
def STUDENT_4555_4561_use_nx_girvan_newman_for_communities(G, graph_layout, node_positions):
    print(
        bcolors.ENDC + "\tCalling routine " + bcolors.HEADER + "STUDENT_4555_4561_use_nx_girvan_newman_for_communities(G,graph_layout,node_positions)" + bcolors.ENDC + "\n")

    start_time = time.time()

    K = []
    k = 1
    comp = girvan_newman(G)
    for communities in itertools.islice(comp, k):
        K = list(tuple(sorted(c) for c in communities))
        print(tuple(sorted(c) for c in communities))

    # my_graph_plot_routine(G, 'r', 'b', 'solid', 'circular', '')

    """
    print(bcolors.ENDC + "\t" + '''
        ######################################################################################################################
        # USE THE BUILT-IN ROUTINE OF NETWORKX FOR CREATING K+1 COMMUNITIES FOR A GRAPH WITH K CONNECTED COMPONENTS, WHERE 
        # THE GIANT COMPONENT IS SPLIT IN TO COMMUNITIES, BY REPEATEDLY REMOVING MAX_BETWEENNESS EDGES, AS SUGGESTED BY THE 
        # GIRVAN-NEWMAN ALGORITHM
        # 
        # INPUT: 
        #   G = the constructed graph to work with. 
        #   graph_layout = a string determining how to position the graph nodes in the plane.
        #   node_positions = a list of previously computed node positions in the plane (for next call of my_graph_plot_routine)
        # OUTPUT: 
        #    community_tuples = a list of K+1 tuples of node_IDs, each tuple determining a different community
        # PSEUDOCODE:    
        #   ...simple...
        ######################################################################################################################
    ''')
    """

    """
    print(bcolors.ENDC + "USEFUL FUNCTIONS:")
    print(bcolors.ENDC + "\t\t" + bcolors.OKCYAN + "girvan_newman(...) "
          + bcolors.ENDC + "from networkx.algorithms.community, provides the requiured list of K+1 community-tuples.\n")
    """

    end_time = time.time()

    print(bcolors.ENDC + "\t===================================================")
    print(bcolors.ENDC + "\tBUILT-IN computation of ONE-SHOT Girvan-Newman clustering for a graph with",
          G.number_of_nodes(), "nodes and", G.number_of_edges(), "links. "
          + "Computation time =", end_time - start_time, "\n")

    return K


global list_of_tuples
list_of_tuples = []


######################################################################################################################
def STUDENT_4555_4561_divisive_community_detection(G, number_of_divisions, graph_layout, node_positions):
    global list_of_tuples
    print(
        bcolors.ENDC + "\tCalling routine " + bcolors.HEADER + "STUDENT_4555_4561_divisive_community_detection(G,number_of_divisions,graph_layout,node_positions)" + bcolors.ENDC + "\n")

    """
    print(bcolors.ENDC + "\t" + '''
        ######################################################################################################################
        # CREATE HIERARCHY OF num_divisions COMMUNITIES FOR A GRAPH WITH num_components CONNECTED COMPONENTS, WHERE 
        # MIN(num_nodes / 4, 10*num_components) >= num_divisions >= K. 
        #
        # INPUT: 
        #   G = the constructed graph to work with. 
        #   graph_layout = a string determining how to position the graph nodes in the plane.
        #   node_positions = a list of previously computed node positions in the plane (for next call of my_graph_plot_routine)
        # OUTPUT: 
        #   A list, hierarchy_of_community_tuples, whose first item is the list community_tuples returned by the 
        #   ONE-SHOT GN algorithm, and each subsequent item is a triple of node-tuples: 
        #    * the tuple of the community to be removed next, and 
        #    * the two tuples of the subcommunities to be added for its substitution
        #
        # PSEUDOCODE:   
        #   HIERARCHY = LIST WITH ONE ITEM, THE community_tuple (LIST OF TUPLES) WITH K+1 COMMUNITIES DETERMINED BY THE 
        # ONE-SHOT GIRVAN-NEWMAN ALGORITHM
        #   REPEAT:
        #       ADD TO THE HIERARCHY A TRIPLE: THE COMMUNITY TO BE REMOVED AND THE TWO SUB-COMMUNITIES THAT MUST SUBSTITUTE IT 
        #       * THE COMMUNITY TO BE REMOVED IS THE LARGEST ONE IN THE PREVIOUS STEP.
        #       * THE TWO SUBCOMMUNITIES TO BE ADDED ARE DETERMINED BY REMOVING MAX_BC EDGES FROM THE SUBGRAPH INDUCED BY THE 
        #       COMMUNITY TO BE REMOVED, UNTIL DISCONNECTION. 
        #   UNTIL num_communities REACHES THE REQUIRED NUMBER num_divisions OF COMMUNITIES
        ######################################################################################################################
    ''')
    """

    start_time = time.time()

    LC = tuple(nx.connected_components(G))
    K = STUDENT_4555_4561_one_shot_girvan_newman_for_communities(G, graph_layout, '')

    for i in range(number_of_divisions - 1):
        # nx.draw(G, with_labels=True)
        # plt.show()
        LC1 = K[0]
        LC2 = K[1]
        list_of_tuples.append([LC, LC1, LC2])
        if len(LC1) > len(LC2):
            G = G.subgraph(LC1)
            G = nx.Graph(G)
            LC = LC1
        else:
            G = G.subgraph(LC2)
            G = nx.Graph(G)
            LC = LC2

        K = STUDENT_4555_4561_one_shot_girvan_newman_for_communities(G, graph_layout, '')

    print("LIST OF TUPLES HIERARCHY: ", list_of_tuples)

    """
    print(bcolors.ENDC + "\tUSEFUL FUNCTIONS:")

    print(bcolors.ENDC + "\t\t"
          + bcolors.OKCYAN + "girvan_newman(...) "
          + bcolors.ENDC + "from networkx.algorithms.community, provides TWO communities for a given CONNECTED graph.")

    print(bcolors.ENDC + "\t\t"
          + bcolors.OKCYAN + "G.subgraph(X) "
          + bcolors.ENDC + "extracts the induced subgraph of G, for a given subset X of nodes.")

    print(bcolors.ENDC + "\t\t"
          + bcolors.OKCYAN + "community_tuples.pop(GiantCommunityIndex) "
          + bcolors.ENDC + "removes the giant community's tuple (to be split) from the community_tuples data structure.")
    print(bcolors.ENDC + "\t\t"
          + bcolors.OKCYAN + "community_tuples.append(...) "
          + bcolors.ENDC + "can add the computed subcommunities' tuples, in substitution of the giant component.")
    """

    end_time = time.time()
    print(bcolors.ENDC + "\t===================================================")
    print(bcolors.ENDC + "\tComputation of HIERARCHICAL BIPARTITION of G in communities, "
          + "using the BUILT-IN girvan-newman algorithm, for a graph with", G.number_of_nodes(), "nodes and",
          G.number_of_edges(), "links. "
          + "Computation time =", end_time - start_time, "\n")

    return list_of_tuples


######################################################################################################################
def STUDENT_4555_4561_determine_opt_community_structure(G, hierarchy_of_community_tuples):
    global list_of_tuples
    print(
        bcolors.ENDC + "\tCalling routine " + bcolors.HEADER + "STUDENT_4555_4561_determine_opt_community_structure(G,hierarchy_of_community_tuples)" + bcolors.ENDC + "\n")

    modularities = []
    LCs = []
    min = 200000
    i = 0
    for com in list_of_tuples:
        p1 = partition(com[1])
        p2 = partition(com[2])
        LCs.append("LC1_" + str((i % 2)))
        LCs.append("LC2_" + str((i % 2)))
        modul1 = community.modularity(p1, com[1])
        modularities.append(modul1)
        modul2 = community.modularity(p2, com[2])
        modularities.append(modul2)
        i += 1

    plt.bar(LCs, modularities)
    plt.title('modularities')
    plt.xlabel('LCs')
    plt.ylabel('modularities')
    plt.show()

    """
    print(bcolors.ENDC + "\t" + '''
        ######################################################################################################################
        # GIVEN A HIERARCHY OF COMMUNITY PARTITIONS FOR A GRAPH, COMPUTE THE MODULARITY OF EAC COMMUNITY PARTITION. 
        # RETURN THE COMMUNITY PARTITION OF MINIMUM MODULARITY VALUE
        # INPUT: 
        #   G = the constructed graph to work with. 
        #   hierarchy_of_community_tuples = the output of the DIVISIVE_COMMUNITY_DETECTION routine
        # OUTPUT: 
        #   The partition which achieves the minimum modularity value, within the hierarchy 
        #        #
        # PSEUDOCODE:
        #   Iterate over the HIERARCHY, to construct (sequentially) all the partitions of the graph into communities
        #       For the current_partition, compute its own current_modularity value
        #   IF      current_modularity_vale < min_modularity_value (so far)
        #   THEN    min_partition = current_partition
        #           min_modularity_value = current_modularity_value
        #   RETURN  (min_partition, min_modularity_value)   
        ######################################################################################################################
        ''')
    """


######################################################################################################################
def STUDENT_4555_4561_add_hamilton_cycle_to_graph(G, node_names_list):
    global graph_nodes_list
    print(
        bcolors.ENDC + "\tCalling routine " + bcolors.HEADER + "STUDENT_4555_4561_add_hamilton_cycle_to_graph(G,node_names_list)" + bcolors.ENDC + "\n")

    hamilton = []
    node_names_list = graph_nodes_list
    for node in range(len(node_names_list)):
        if node + 1 < len(node_names_list):
            G.add_edge(node_names_list[node], node_names_list[node + 1])
            hamilton.append((node_names_list[node], node_names_list[node + 1]))
        else:
            G.add_edge(node_names_list[0], node_names_list[node])
            hamilton.append((node_names_list[0], node_names_list[node]))

    print("HAMILTON LIST: ", hamilton)
    my_graph_plot_routine(G, 'r', 'b', 'solid', 'circular', '')

    """print(bcolors.ENDC + "\t" + '''
        ######################################################################################################################
        # GIVEN AN INPUT GRAPH WHICH IS DISCONNECTED, ADD A (HAMILTON) CYCLE CONTAINING ALL THE NODES IN THE GRAPH
        # INPUT: A graph G, and a list of node-IDs.
        # OUTPUT: The augmented graph of G, with a (MAMILTON) cycle containing all the nodes from the node_names_list (in that order).
        #
        # COMMENT: The role of this routine is to add a HAMILTON cycle, i.e., a cycle that contains all the nodes in the graph.
        # Such an operation will guarantee that the graph is connected and, moreover, there is no bridge in the new graph.
        ######################################################################################################################
    ''')

    print(bcolors.ENDC + "\tUSEFUL FUNCTIONS:")

    print("\t\t"
          + bcolors.OKCYAN + "my_graph_plot_routine(G,'grey','blue','solid',graph_layout,node_positions) "
          + bcolors.ENDC + "plots the graph with the grey-color for nodes, and (blue color,solid style) for the edges.\n")
    """

    return G


######################################################################################################################
# ADD RANDOM EDGES TO A GRAPH...
######################################################################################################################
def STUDENT_4555_4561_add_random_edges_to_graph(G, node_names_list, NUM_RANDOM_EDGES, EDGE_ADDITION_PROBABILITY):
    global graph_nodes_list
    print(bcolors.ENDC + "\tCalling routine "
          + bcolors.HEADER + "STUDENT_4555_4561_add_random_edges_to_graph(G,node_names_list,NUM_RANDOM_EDGES,EDGE_ADDITION_PROBABILITY)"
          + bcolors.ENDC + "\n")

    loopless_links_list = G.edges
    node_names_list = graph_nodes_list
    # initialize neighbors dictionary
    neighbors_of = dict()
    not_neighbors_of = dict()
    for i in node_names_list:
        neighbors_of[i] = []
        not_neighbors_of[i] = []

    print("NODE NAME LIST: ", node_names_list)
    print("LOOPLESS: ", loopless_links_list)
    # add all of the neighbors of every node to a list in dictionary
    for tup in loopless_links_list:
        neighbors_of[tup[0]].append(tup[1])
        neighbors_of[tup[1]].append(tup[0])
        neighbors_of[tup[0]] = list(set(neighbors_of[tup[0]]))
        neighbors_of[tup[1]] = list(set(neighbors_of[tup[1]]))

    # get list of every node non-neighbors
    for i in node_names_list:
        for key in neighbors_of:
            if i != key:
                if i not in neighbors_of[key]:
                    not_neighbors_of[key].append(i)

    # Add random (X,Y) edges
    for X in node_names_list:
        for i in range(NUM_RANDOM_EDGES):
            if len(not_neighbors_of[X]) > 0:
                Y = random.choice(not_neighbors_of[X])
                # random coin-flip(select) a number between [0,1]
                coin_flip = random.uniform(0, 1)
                if coin_flip > EDGE_ADDITION_PROBABILITY:
                    G.add_edge(X, Y)
                    not_neighbors_of[X].remove(Y)

    # Print the graph with new edges
    my_graph_plot_routine(G, 'r', 'b', 'solid', 'circular', '')

    """
    print(bcolors.ENDC + "\t" + '''
        ######################################################################################################################
        # GIVEN AN INPUT GRAPH WHICH IS DISCONNECTED, ADD A (HAMILTON) CYCLE CONTAINING ALL THE NODES IN THE GRAPH
        # INPUT: A graph G, an integer indicating the max number of random edges to be added per node, and an edge addition 
        # probability, for each random edge considered. 
        # 
        # OUTPUT: The augmented graph of G, containing also  all the newly added random edges.
        #
        # COMMENT: The role of this routine is to add, per node, a number of random edges to other nodes.
        # Create a for-loop, per node X, and then make NUM_RANDOM_EDGES attempts to add a new random edge from X to a 
        # randomly chosen destination (outside its neighborhood). Each attempt will be successful with probability 
        # EDGE_ADDITION_PROBABILITY (i.e., only when a random coin-flip in [0,1] returns a value < EDGE_ADDITION_PROBABILITY).")
        ######################################################################################################################
    ''')
    
    print(bcolors.ENDC + "\tUSEFUL FUNCTIONS:")
    print("\t\t"
          + bcolors.OKCYAN + "my_graph_plot_routine(G,'grey','blue','solid',graph_layout,node_positions) "
          + bcolors.ENDC + "plots the graph with the grey-color for nodes, and (blue color,solid style) for the edges.\n")
    """

    return G


######################################################################################################################
# VISUALISE COMMUNITIES WITHIN A GRAPH
######################################################################################################################
def STUDENT_4555_4561_visualize_communities(G, community_tuples, graph_layout, node_positions):
    global graph_nodes_list
    print(bcolors.ENDC + "\tCalling routine "
          + bcolors.HEADER + "STUDENT_4555_4561_visualize_communities(G,community_tuples,graph_layout,node_positions)" + bcolors.ENDC + "\n")

    node_names_list = graph_nodes_list

    # plot the communities
    community_tuples = STUDENT_4555_4561_one_shot_girvan_newman_for_communities(G, graph_layout, '')
    print(community_tuples)
    #colors = my_random_color_list_generator(len(community_tuples))
    color_map = []

    colors = ["lightcoral", "gray", "lightgray", "firebrick", "red", "chocolate", "darkorange", "moccasin", "gold",
              "yellow", "darkolivegreen", "chartreuse", "forestgreen", "lime", "mediumaquamarine", "turquoise", "teal",
              "cadetblue", "dogerblue", "blue", "slateblue", "blueviolet", "magenta", "lightsteelblue"]
    for com in community_tuples:
        color = random.choice(colors)
        for node in com:
            position = node_names_list.index(node)
            color_map.insert(position, color)
        colors.remove(color)

    """
    for com in community_tuples:
        color = random.choice(colors)
        for node in node_names_list:
            pos = node_names_list.index(node)
            print("NODE: ", node ,"IN POSITION: ", pos)
            color_map.insert(pos, color)
            print("COLOR MAP: ", color_map)
        colors.remove(color)
        print("COLORS:", colors)
    """
    nx.draw(G, node_color=color_map, with_labels=True)
    plt.show()

    """
    print(
        bcolors.ENDC + "\tINPUT: A graph G, and a list of lists/tuples each of which contains the nodes of a different community.")
    print(
        bcolors.ENDC + "\t\t The graph_layout parameter determines how the " + bcolors.OKCYAN + "my_graph_plot_routine will position the nodes in the plane.")
    print(
        bcolors.ENDC + "\t\t The node_positions list contains an existent placement of the nodes in the plane. If empty, a new positioning will be created by the " + bcolors.OKCYAN + "my_graph_plot_routine" + bcolors.ENDC + ".\n")

    print(bcolors.ENDC + "\tOUTPUT: Plot the graph using a different color per community.\n")

    print(bcolors.ENDC + "\tUSEFUL FUNCTIONS:")
    print(
        bcolors.OKCYAN + "\t\tmy_random_color_list_generator(number_of_communities)" + bcolors.ENDC + " initiates a list of random colors for the communities.")
    print(
        bcolors.OKCYAN + "\t\tmy_graph_plot_routine(G,node_colors,'blue','solid',graph_layout,node_positions)" + bcolors.ENDC + " plots the graph with the chosen node colors, and (blue color,solid style) for the edges.")
    """


########################################################################################
########################### STUDENT_4555_4561 ROUTINES LIBRARY ENDS ###########################
########################################################################################


########################################################################################
############################# MAIN MENUS OF USER CHOICES ############################### 
########################################################################################

############################### GLOBAL INITIALIZATIONS #################################

global node_names_list
G = nx.Graph()  # INITIALIZATION OF THE GRAPH TO BE CONSTRUCTED
node_names_list = []
node_positions = []  # INITIAL POSITIONS OF NODES ON THE PLANE ARE UNDEFINED...
community_tuples = []  # INITIALIZATION OF LIST OF COMMUNITY TUPLES...
hierarchy_of_community_tuples = []  # INITIALIZATION OF HIERARCHY OF COMMUNITY TUPLES

G, node_names_list, node_positions = my_menu_graph_construction(G, node_names_list, node_positions)

my_menu_community_detection(G, node_names_list, node_positions, hierarchy_of_community_tuples)
