"""
Graph Visualization Module for Scratch Knowledge Graph App
"""
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging

from config import MAX_GRAPH_NODES, GRAPH_LAYOUT, GRAPH_FIGSIZE

logger = logging.getLogger(__name__)

class GraphVisualizer:
    """Graph visualization using NetworkX and Matplotlib"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.node_colors = []
        self.edge_colors = []
    
    def create_graph(self, entities: pd.DataFrame, relationships: pd.DataFrame) -> nx.Graph:
        """Create NetworkX graph from entities and relationships"""
        self.graph.clear()
        
        # Add nodes (entities)
        for _, entity in entities.iterrows():
            if len(self.graph.nodes) >= MAX_GRAPH_NODES:
                break
            
            node_id = entity['id']
            node_label = entity.get('title', entity.get('name', node_id))
            self.graph.add_node(
                node_id,
                label=node_label,
                description=entity.get('description', ''),
                type='entity'
            )
        
        # Add edges (relationships)
        for _, rel in relationships.iterrows():
            source = rel['source']
            target = rel['target']
            
            # Only add edge if both nodes exist
            if source in self.graph.nodes and target in self.graph.nodes:
                self.graph.add_edge(
                    source,
                    target,
                    label=rel.get('description', ''),
                    weight=rel.get('weight', 1.0),
                    type='relationship'
                )
        
        return self.graph
    
    def visualize(self, entities: pd.DataFrame, relationships: pd.DataFrame, 
                  title: str = "Knowledge Graph") -> plt.Figure:
        """Create and return matplotlib figure with graph visualization"""
        try:
            # Create the graph
            graph = self.create_graph(entities, relationships)
            
            if len(graph.nodes) == 0:
                # Create empty figure with message
                fig, ax = plt.subplots(figsize=GRAPH_FIGSIZE)
                ax.text(0.5, 0.5, 'Không có dữ liệu để hiển thị graph', 
                       ha='center', va='center', fontsize=16, 
                       transform=ax.transAxes)
                ax.set_title(title, fontsize=18, fontweight='bold')
                ax.axis('off')
                return fig
            
            # Create figure
            fig, ax = plt.subplots(figsize=GRAPH_FIGSIZE)
            
            # Choose layout
            if GRAPH_LAYOUT == "spring":
                pos = nx.spring_layout(graph, k=3, iterations=50)
            elif GRAPH_LAYOUT == "circular":
                pos = nx.circular_layout(graph)
            elif GRAPH_LAYOUT == "hierarchical":
                pos = nx.hierarchical_layout(graph)
            else:
                pos = nx.spring_layout(graph)
            
            # Prepare colors
            node_colors = self._get_node_colors(graph)
            edge_colors = self._get_edge_colors(graph)
            
            # Draw nodes
            nx.draw_networkx_nodes(
                graph, pos,
                node_color=node_colors,
                node_size=1000,
                alpha=0.8,
                ax=ax
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                graph, pos,
                edge_color=edge_colors,
                width=2,
                alpha=0.6,
                ax=ax
            )
            
            # Draw labels
            labels = nx.get_node_attributes(graph, 'label')
            nx.draw_networkx_labels(
                graph, pos,
                labels=labels,
                font_size=10,
                font_weight='bold',
                ax=ax
            )
            
            # Draw edge labels (optional, can be cluttered)
            if len(graph.edges) <= 10:  # Only show edge labels for small graphs
                edge_labels = nx.get_edge_attributes(graph, 'label')
                nx.draw_networkx_edge_labels(
                    graph, pos,
                    edge_labels=edge_labels,
                    font_size=8,
                    ax=ax
                )
            
            # Customize appearance
            ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
            ax.axis('off')
            
            # Add legend
            self._add_legend(ax)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating graph visualization: {e}")
            return self._create_error_figure(str(e))
    
    def _get_node_colors(self, graph: nx.Graph) -> list:
        """Get colors for nodes based on their properties"""
        colors = []
        color_map = {
            'entity': '#4CAF50',      # Green
            'concept': '#2196F3',     # Blue
            'action': '#FF9800',      # Orange
            'object': '#9C27B0',     # Purple
            'default': '#607D8B'      # Blue Grey
        }
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            node_type = node_data.get('type', 'default')
            colors.append(color_map.get(node_type, color_map['default']))
        
        return colors
    
    def _get_edge_colors(self, graph: nx.Graph) -> list:
        """Get colors for edges based on their properties"""
        colors = []
        color_map = {
            'relationship': '#757575',  # Grey
            'hierarchy': '#E91E63',     # Pink
            'association': '#00BCD4',   # Cyan
            'default': '#9E9E9E'        # Light Grey
        }
        
        for edge in graph.edges():
            edge_data = graph.edges[edge]
            edge_type = edge_data.get('type', 'default')
            colors.append(color_map.get(edge_type, color_map['default']))
        
        return colors
    
    def _add_legend(self, ax):
        """Add legend to the graph"""
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='#4CAF50', markersize=10, label='Thực thể'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='#2196F3', markersize=10, label='Khái niệm'),
            plt.Line2D([0], [0], color='#757575', linewidth=2, label='Mối quan hệ')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1, 1), fontsize=10)
    
    def _create_error_figure(self, error_message: str) -> plt.Figure:
        """Create error figure when visualization fails"""
        fig, ax = plt.subplots(figsize=GRAPH_FIGSIZE)
        ax.text(0.5, 0.5, f'Lỗi khi tạo graph:\n{error_message}', 
               ha='center', va='center', fontsize=14, 
               transform=ax.transAxes, color='red')
        ax.set_title('Graph Visualization Error', fontsize=18, fontweight='bold')
        ax.axis('off')
        return fig
    
    def get_graph_stats(self, entities: pd.DataFrame, relationships: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the graph"""
        graph = self.create_graph(entities, relationships)
        
        stats = {
            'nodes': len(graph.nodes),
            'edges': len(graph.edges),
            'density': nx.density(graph),
            'connected_components': nx.number_connected_components(graph),
            'average_clustering': nx.average_clustering(graph) if len(graph.nodes) > 0 else 0
        }
        
        return stats

# Global instance
graph_visualizer = GraphVisualizer()



