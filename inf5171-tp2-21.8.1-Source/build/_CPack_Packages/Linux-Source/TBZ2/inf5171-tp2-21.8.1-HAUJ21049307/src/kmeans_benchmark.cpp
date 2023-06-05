#include "kmeans.h"
#include "kmeans_parallel.h"
#include "modules.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
double meanVector(std::vector<double> v){
double mean = 0;
  for (size_t i = 0; i < v.size(); i++){
  mean += v[i];
  }
  mean = mean/v.size();
  return mean;
}
double roundNumber(double n){
  return ( std::ceil(n * 100.0)/100.0);
}
int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "missing argument: problem file" << std::endl;
  }

  std::vector<KMeansClusterDefinition> definitions(KMeansClusterDefinition::load(argv[1]));
  if (definitions.size() == 0) {
    std::printf("failed to open problem file %s\n", argv[1]);
    return -1;
  }
  // Écriture du fichier benchmark.csv

  std::ofstream file;
  file.open("./../build/benchmark.csv");
  file << "Temps série, Temps parallèle, Accélération, Efficience\n";
  size_t donnée = 1;

  std::vector<double> tempsSerie ;
  std::vector<double> tempsParallel ;
  std::vector<double> acceleration;
  double efficience = 0;

  // Boucle  pour faire 10 ensembles de données différents
  for ( size_t i = 0 ; i < 10; i++ ){
    std::cout << "Ensemble de donnée " << donnée++ << std::endl;
    DataSet training_set;
    training_set.generate_random(definitions);

    DataSet eval_set;
    eval_set.generate_random(definitions);
    // Points de départ pour initialiser l'algorithme
    std::vector<Point> start_points
      = start_points_random(training_set.m_points, definitions.size(), 0);
    for (const struct test_klass& item : algo) {
      std::cout << "BENCHMARK " << item.klass << "\n";
      std::shared_ptr<KMeans> kmeans = item.factory();
      // Trouver les groupes dans le jeu de données
      const auto t1 = std::chrono::steady_clock::now();
      kmeans->fit(training_set.m_points, start_points);

      // classifier un nouveau jeu de données
      std::vector<uint8_t> eval_cluster;
      kmeans->classify(eval_set.m_points, eval_cluster);

      // comparer pour obtenir le taux de précision
      std::unordered_map<size_t, size_t> mapping;
      make_cluster_mapping(kmeans->get_centers(), definitions, mapping);

      size_t good = kmeans->compare(eval_set.m_cluster, eval_cluster, mapping);
      const auto t2 = std::chrono::steady_clock::now();
      std::cout << "precision: " << 100 * good / eval_cluster.size() << " %" << std::endl;
      double time =  roundNumber(std::chrono::duration<double>(t2 - t1).count());
      std::cout  << "Time = " <<  time  << std::endl;
      if( item.klass == "serial" ){
        tempsSerie.push_back(time);
      } else if ( item.klass == "parallel" ){
        tempsParallel.push_back(time);
      }
    }
    acceleration.push_back( roundNumber( tempsSerie[i] / tempsParallel[i] ) ) ;
    efficience = roundNumber(acceleration[i]/std::thread::hardware_concurrency());

    file << tempsSerie[i] << " " << tempsParallel[i] << " " << acceleration[i] << " " << efficience << "\n";
  }
    double moyenneSerie = roundNumber( meanVector(tempsSerie) );
    double moyenneParallel = roundNumber( meanVector(tempsParallel) );
    double moyenneAcceleration = roundNumber( meanVector(acceleration) );
    file << "Moyenne temps série : " <<  moyenneSerie << "\nMoyenne temps parallèle : " << moyenneParallel << "\nMoyenne accélération : " << moyenneAcceleration; 

  file.close();
  return 0;
}
