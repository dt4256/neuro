#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <ctime>

namespace reg {

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
				for (int j = 0; j < (int)network[i].size(); j++) {
					for (int prev_j = 0; prev_j < (int)network[i - 1].size(); prev_j++) {
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
			layers = (int)layer_sizes.size();
			for (int i = 0; i < layers; i++) {
				std::vector<neuron> layer;
				for (int j = 0; j < layer_sizes[i]; j++) {
					layer.push_back(neuron{ 0.0, 0.0, 0.0, 0.0, {} });
				}
				network.push_back(layer);
			}
			get_ready();
		}

		void get_ready_for_new() {
			for (int i = 1; i < layers; i++) {
				for (int j = 0; j < (int)network[i].size(); j++) {
					network[i][j].delta = 0.0;
				}
			}
		}

		void initialize_first_layer(std::vector<double> input) {
			if (input.size() != network[0].size()) return;
			for (int j = 0; j < (int)network[0].size(); j++) {
				network[0][j].v_out = input[j];
			}
		}

		void count_layer(int index) {
			if (index <= 0 || index >= layers) return;

			for (int j = 0; j < (int)network[index].size(); j++) {
				double sum = 0.0;
				for (int k = 0; k < (int)network[index][j].connections.size(); k++) {
					int p_j = network[index][j].connections[k].from_j;
					double w = network[index][j].connections[k].weight;
					sum += network[index - 1][p_j].v_out * w;
				}

				network[index][j].v_in = sum + network[index][j].b;

				// Логика регрессии: на последнем слое нет сигмоиды
				if (index == layers - 1) network[index][j].v_out = network[index][j].v_in;
				else network[index][j].v_out = sigmoid(network[index][j].v_in);
			}
		}

		void count_all() {
			for (int i = 1; i < layers; i++) {
				count_layer(i);
			}
		}

		void count_last_delta(std::vector<double> waited) {
			int t1 = (int)network[layers - 1].size();
			if (t1 != (int)waited.size()) return;
			for (int i = 0; i < (int)network[layers - 1].size(); i++) {
				// Дельта для регрессии (производная линейной функции = 1)
				network[layers - 1][i].delta = waited[i] - network[layers - 1][i].v_out;
			}
		}

		void count_all_deltas(std::vector<double> waited) {
			get_ready_for_new();
			count_last_delta(waited);
			for (int i = layers - 2; i > 0; i--) {
				for (int next_i = 0; next_i < (int)network[i + 1].size(); ++next_i) {
					for (int j = 0; j < (int)network[i + 1][next_i].connections.size(); ++j) {
						connection t = network[i + 1][next_i].connections[j];
						network[i][t.from_j].delta += network[i + 1][next_i].delta * t.weight;
					}
				}
				for (int j = 0; j < (int)network[i].size(); ++j) {
					network[i][j].delta *= proizvod(network[i][j].v_out);
				}
			}
		}

		void update_weights() {
			for (int i = 1; i < layers; i++) {
				for (int j = 0; j < (int)network[i].size(); j++) {
					network[i][j].b += network[i][j].delta * lr;
					for (int k = 0; k < (int)network[i][j].connections.size(); k++) {
						int prev_j = network[i][j].connections[k].from_j;
						network[i][j].connections[k].weight += network[i][j].delta * network[i - 1][prev_j].v_out * lr;
					}
				}
			}
		}

		double proizvod(double v_out) {
			return v_out * (1 - v_out);
		}

		double sigmoid(double v_in) {
			return 1 / (1 + exp(-v_in));
		}
	};

	

} // namespace nn