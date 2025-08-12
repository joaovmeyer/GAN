#include <iostream>
#include <cmath>

#define NUM_TYPE float

#include "../graph.h"
#include "operations.hpp"
#include "optimizer.hpp"
#include "../rng.h"

using namespace std;






Vec smoothLeakyReLU(const Vec& x, float alpha = 0.02) {
	return log(exp(x * (1.0f - alpha)) + 1.0f) + x * alpha;
}

Vec ReLU(const Vec& x) {
	Vec zero = Vector::build(x->size, 0.0f);
	return max(zero, x);
}

Vec LeakyReLU(const Vec& x, float alpha = 0.02) {
	return max(x * alpha, x);
}






int main() {

	rng::setSeed(42);

	// models definition
	int hidden_size = 24;
	int noise_size = 2;

	// generator
	Mat G_W1 = Matrix::makeRandom(hidden_size, noise_size, 0.0, 1.0, true);
	Mat G_W2 = Matrix::makeRandom(2, hidden_size, 0.0, 0.3, true);
	Vec G_b1 = Vector::build(hidden_size, 0.1, true);
	Vec G_b2 = Vector::build(2, 0.1, true);
	std::vector<std::shared_ptr<Node>> G_parameters = { G_W1, G_W2, G_b1, G_b2 };

	auto G = [&](const Vec& x) {
		// first layer
		Vec L1 = LeakyReLU(G_W1 * x + G_b1);

		// second layer
		return tanh(G_W2 * L1 + G_b2);

	};

	// discriminator
	Mat D_W1 = Matrix::makeRandom(hidden_size, 2, 0.0, 1.0, true);
	Mat D_W2 = Matrix::makeRandom(1, hidden_size, 0.0, 0.3, true);
	Vec D_b1 = Vector::build(hidden_size, 0.1, true);
	Vec D_b2 = Vector::build(1, 0.1, true);
	std::vector<std::shared_ptr<Node>> D_parameters = { D_W1, D_W2, D_b1, D_b2 };

	auto D = [&](const Vec& x) {
		// first layer
		Vec L1 = LeakyReLU(D_W1 * x + D_b1);

		// second layer
		return sigmoid(D_W2 * L1 + D_b2)[0];

	};


	// distributions definition
	float radius = 0.5f;

	auto data_distribution = [&]() {
		float theta = rng::fromUniformDistribution(0.0, 2.0 * 3.1415926535);

		float noise = rng::fromNormalDistribution(0.0, 0.03);
		float x = std::cos(theta) * (radius + noise);
		float y = std::sin(theta) * (radius + noise);

		Vec v = Vector::build(2); v->value = { x, y };
		return v;
	};

	auto noise_distribution = [&]() {
		Vec v = Vector::build(noise_size);
		for (int i = 0; i < noise_size; ++i) {
			v->value[i] = rng::fromUniformDistribution(-1.0, 1.0);
		}
		return v;

	};



	// plot data distribution
	Graph graph{};
	for (int i = 0; i < 10000; ++i) {
		Vec x = data_distribution();

		graph.addPoint(Point(x->value[0], x->value[1], 3, olc::RED));
	}

	graph.waitFinish();



	// optimizers
	Optimizer<Adam> G_optimizer(G_parameters, 0.0005, 0.5, 0.999);
	Optimizer<Adam> D_optimizer(D_parameters, 0.0005, 0.5, 0.999);



	// training (https://www.researchgate.net/publication/263012109_Generative_Adversarial_Networks)
	int batch_size = 48;
	int n_iter = 50000;
	int k = 2;

	for (int i = 0; i < n_iter; ++i) {

		// train discriminator for k steps
		for (int j = 0; j < k; ++j) {
			Var D_loss = Scalar::build(0.0f);

			for (int l = 0; l < batch_size; ++l) {
				Vec x_l = data_distribution();
				Vec z_l = noise_distribution();
				D_loss = D_loss + log(D(x_l) + 1e-4) + log(1.0f - D(G(z_l)) + 1e-4);
			}

			// negative because we want to maximize
			D_loss = -D_loss / batch_size;

			// update discriminator parameters
			D_optimizer.prepare();
			D_loss->calculateDerivatives();
			D_optimizer.step();
		}

		// train generator once

		Var G_loss = Scalar::build(0.0f);
		for (int l = 0; l < batch_size; ++l) {
			Vec z_l = noise_distribution();
			// uses non-saturating loss, that results in stronger gradients early on training
			G_loss = G_loss + log(D(G(z_l)) + 1e-4);
		}

		G_loss = -G_loss / batch_size;

		// update generator parameters
		G_optimizer.prepare();
		G_loss->calculateDerivatives();
		G_optimizer.step();


		// show results every so often
		if ((i + 1) % (n_iter / 10) == 0) {

			Graph graph{};
			int n_points = 1000;

			for (int j = 0; j < n_points; ++j) {
				Vec z = noise_distribution();
				Vec x = data_distribution();

				std::vector<float> res = G(z)();

				graph.addPoint(Point(x->value[0], x->value[1], 3, olc::RED));
				graph.addPoint(Point(res[0], res[1], 3, olc::BLUE));
			}

			graph.waitFinish();

		}
	}


	return 0;
}
