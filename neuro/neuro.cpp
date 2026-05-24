#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <windows.h>
#include "RegressionNetwork.hpp"

std::vector<std::vector<double>> loadCSV(const std::string& filename, bool hasHeader = true) {
	std::vector<std::vector<double>> data;
	std::ifstream file(filename);

	if (!file.is_open()) {
		std::cerr << "Error: Could not open file " << filename << std::endl;
		return data;
	}

	std::string line;

	// Пропускаем заголовок
	if (hasHeader) {
		std::getline(file, line);
	}

	while (std::getline(file, line)) {
		// Очистка от спецсимволов Windows (\r)
		line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

		if (line.empty()) continue;

		std::vector<double> row;
		std::stringstream ss(line);
		std::string value;

		// Читаем значения, разделенные запятой
		while (std::getline(ss, value, ',')) {
			try {
				// Преобразуем в double (std::stod)
				row.push_back(std::stod(value));
			}
			catch (...) {
				// Если попался текст или пустая ячейка — ставим 0.0
				row.push_back(0.0);
			}
		}

		data.push_back(row);
	}

	file.close();
	return data;
}

int main()
{
	SetConsoleCP(65001);
	SetConsoleOutputCP(65001);
	std::ios_base::sync_with_stdio(false);
	srand(time(NULL));
	std::vector<std::vector<double>> X_train = loadCSV("X_train.csv");
	std::vector<std::vector<double>> y_train = loadCSV("y_train.csv");
	std::vector<std::vector<double>> X_test = loadCSV("X_test.csv");
	std::vector<std::vector<double>> y_test = loadCSV("y_test.csv");
	reg::neuronetwork net({(int)X_train[0].size(),32,16,1}, 0.001);
	auto startlr = std::chrono::high_resolution_clock::now();
	for (int epoch = 0; epoch < 1000; epoch++) {
		if (epoch == 500)net.lr = 0.0001;
		for (int i = 0; i < X_train.size(); i++) {
			net.initialize_first_layer(X_train[i]);
			net.count_all();
			net.count_all_deltas(y_train[i]);
			net.update_weights();
		}

		if (epoch % 10 == 0) {
			int counter = 0;double sum = 0;
			for (int i = 0; i < X_test.size(); ++i) {
				net.initialize_first_layer(X_test[i]);
				net.count_all();
				counter++;
				sum += (net.network.back()[0].v_out - y_test[i][0]) * (net.network.back()[0].v_out - y_test[i][0]);
			}std::cout<<"Epoch: "<<epoch<<", MSE: "<<sum / counter<<'\n';
		}
		if (epoch == 999) {
			for (int i = 0; i < X_test.size(); ++i) {
				net.initialize_first_layer(X_test[i]);
				net.count_all();
				std::cout << "Waited: " << y_test[i][0] << " Obtained: " << net.network.back()[0].v_out << '\n';
			}
		}
	}
	auto endlr = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> durationlr = endlr - startlr;
	std::cout << "Learning time: " << durationlr.count() << " seconds\n";
	std::cout << "Training completed. The computer will hibernate in 5 seconds..." << std::endl;

	// Засыпаем на 5 секунд
	std::this_thread::sleep_for(std::chrono::seconds(5));

	// Команда гибернации
	system("shutdown /h");
	return 0;
}

