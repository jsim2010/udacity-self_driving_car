/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
static default_random_engine random_engine;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    num_particles = 90;
    
    normal_distribution<double> x_distribution(x, std[0]);
    normal_distribution<double> y_distribution(y, std[1]);
    normal_distribution<double> theta_distribution(theta, std[2]);

    // Initialize each particle.
    for (int id = 0; id < num_particles; ++id) {
        Particle particle;

        particle.id = id;
        particle.x = x_distribution(random_engine);
        particle.y = y_distribution(random_engine);
        particle.theta = theta_distribution(random_engine);
        particle.weight = 1.0;

        particles.push_back(particle);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    for (int i = 0; i < num_particles; ++i) {
        double theta = particles[i].theta;

        if (fabs(yaw_rate) < 0.001) {
            particles[i].x += velocity * delta_t * cos(theta);
            particles[i].y += velocity * delta_t * sin(theta);
        }
        else {
            particles[i].x += (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
            particles[i].y += (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t));
        }

        particles[i].theta += yaw_rate * delta_t;

        normal_distribution<double> x_distribution(particles[i].x, std_pos[0]);
        normal_distribution<double> y_distribution(particles[i].y, std_pos[1]);
        normal_distribution<double> theta_distribution(particles[i].theta, std_pos[2]);
    
        particles[i].y = y_distribution(random_engine);
        particles[i].x = x_distribution(random_engine);
        particles[i].theta = theta_distribution(random_engine);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    for (unsigned int o = 0; o < observations.size(); ++o) {
        double nearest_distance = numeric_limits<double>::infinity();

        for (unsigned int p = 0; p < predicted.size(); ++p) {
            double distance = dist(observations[o].x, observations[o].y, predicted[p].x, predicted[p].y);

            if (distance < nearest_distance) {
                nearest_distance = distance;
                observations[o].id = predicted[p].id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    for (int i = 0; i < num_particles; ++i) {
        std::vector<LandmarkObs> landmarks(map_landmarks.landmark_list.size());
        std::vector<LandmarkObs> particle_observations(observations);
        double xp = particles[i].x;
        double yp = particles[i].y;
        double theta = particles[i].theta;

        // Build landmarks.
        for (Map::single_landmark_s map_landmark : map_landmarks.landmark_list) {
            int id = map_landmark.id_i - 1;
            landmarks[id].id = id;
            landmarks[id].x = map_landmark.x_f;
            landmarks[id].y = map_landmark.y_f;
        }

        // Transform observations.
        for (unsigned int j = 0; j < particle_observations.size(); ++j) {
            double x = observations[j].x;
            double y = observations[j].y;
            particle_observations[j].x = x * cos(theta) - y * sin(theta) + xp;
            particle_observations[j].y = x * sin(theta) + y * cos(theta) + yp;
        }

        dataAssociation(landmarks, particle_observations);
        
        particles[i].weight = 1.0;

        for (LandmarkObs observation : particle_observations) {
            LandmarkObs landmark = landmarks[observation.id];
            double sig_x = std_landmark[0];
            double sig_y = std_landmark[1];
            double gaussian_norm = (1.0 / (2.0 * M_PI * sig_x * sig_y));
            double exponent = pow(observation.x - landmark.x, 2) / (2 * pow(sig_x, 2)) + pow(observation.y - landmark.y, 2) / (2 * pow(sig_y, 2));

            particles[i].weight *= gaussian_norm * exp(-exponent);
        }
    }
}

void ParticleFilter::resample() {
    std::vector<double> weights;
    std::vector<Particle> new_particles;

    for (Particle particle : particles) {
        weights.push_back(particle.weight);
    }

    std::discrete_distribution<> distribution(weights.begin(), weights.end());
    default_random_engine random_engine;

    for (int i = 0; i < num_particles; ++i) {
        new_particles.push_back(particles[distribution(random_engine)]);
    }

    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
