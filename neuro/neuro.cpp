#include <iostream>
#include <vector>
#include <map>
#include <cmath>

struct connection
{
	int from_j;
	double weight;
};

struct neuron {
	double b;
	double v_out;
	double v_in;
	double delta;
	std::vector<connection> connections;

};


class neuronetwork {
	private:
		int layers;
		double lr;
		

		void get_ready() {
			for (int i = 1; i < layers; i++) {
				for (int j = 0; j < network[i].size(); j++) {
					for (int prev_j = 0; prev_j < network[i - 1].size(); prev_j++) {
						double rand_w = ((double)rand() / RAND_MAX) * 2 - 1;
						network[i][j].connections.push_back(connection{ prev_j, rand_w });
					}
				}
			}
		}
	public:
		std::vector<std::vector<neuron>> network;
		neuronetwork(std::vector<int> layer_sizes, double lr = 0.0006) {
			this->lr = lr;
			layers = layer_sizes.size();
			for (int i = 0; i < layers; i++) {
				std::vector<neuron> layer;
				for (int j = 0; j < layer_sizes[i]; j++) {
					layer.push_back(neuron{ 0.0, 0.0, 0.0, {} });
				}
				network.push_back(layer);
			}
			get_ready();
		}
		
		void initialize_first_layer(std::vector<double> input) {
			if (input.size() != network[0].size()) return;
			for (int j = 0; j < network[0].size(); j++) {
				network[0][j].v_out = input[j];
			}
		}
		void count_layer(int index) {
			if (index <= 0 || index >= layers) return;

			for (int j = 0; j < network[index].size(); j++) {
				double sum = 0.0;
				for (int k = 0; k < network[index][j].connections.size(); k++) {
					int p_j = network[index][j].connections[k].from_j;
					double w = network[index][j].connections[k].weight;

					sum += network[index - 1][p_j].v_out * w;
				}

				network[index][j].v_in = sum + network[index][j].b;
				network[index][j].v_out = sigmoid(network[index][j].v_in);
			}
		}
		void count_all() {
			for (int i = 1; i < layers; i++) {
				count_layer(i);
			}
		}

		



		double sigmoid(double v_in)
		{
			return 1 / (1 + exp(-v_in));
		}

};


int main()
{
	
}

