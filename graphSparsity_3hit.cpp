#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <map>
#include <vector>
#include <set>
#include <algorithm>
#include <chrono>

#define MAX_BUF_SIZE 1024

struct Gene {
	std::set<int> tumor;
	std::set<int> normal;
};


void funcName(std::vector<Gene> data, int totalGenes, int numTumor, int numNormal){
	long long int count = 0;
	for (int i = 0; i < totalGenes; i++){
		Gene gene1 = data[i];
		for (int j = i + 1; j < totalGenes; j++){
			Gene gene2 = data[j];
			std::set<int> intersectNormal1, intersectTumor1;
		
			std::set_intersection(gene1.tumor.begin(), gene1.tumor.end(), gene2.tumor.begin(), gene2.tumor.end(), std::inserter(intersectTumor1, intersectTumor1.begin()));	
			std::set_intersection(gene1.normal.begin(), gene1.normal.end(), gene2.normal.begin(), gene2.normal.end(), std::inserter(intersectNormal1, intersectNormal1.begin()));
			
			if (!intersectTumor1.empty()) {
				for (int k = j + 1; k < totalGenes; k++){
					
					Gene gene3 = data[k];
					std::set<int> intersectNormal2, intersectTumor2;
					std::set_intersection(gene3.tumor.begin(), gene3.tumor.end(), intersectTumor1.begin(), intersectTumor1.end(), std::inserter(intersectTumor2, intersectTumor2.begin()));	
					std::set_intersection(gene3.normal.begin(), gene3.normal.end(), intersectNormal1.begin(), intersectNormal1.end(), std::inserter(intersectNormal2, intersectNormal2.begin()));


					if (!intersectTumor2.empty() && intersectNormal2.empty()){
						count++;

						if (count % 10000 == 0){
							printf("We are at %lld combinations with genes %d %d %d\n", count, i, j, k);
						}
					
					}
							
				}
			}
		}
	}

	printf("Number of combinations: %lld\n", count);

		
}
	

int main(int argc, char *argv[]){


	if (argc != 2){
		printf("One argument expected: ./graphSparcity <dataFile>");
	}

	FILE* dataFile;
	dataFile = fopen(argv[1], "r");
	
	if (dataFile == NULL) {
		perror("Error opening file");
		return 1; 
	}
	char geneId[MAX_BUF_SIZE], sampleId[MAX_BUF_SIZE];
	int numGenes, numSamples, value, numTumor, numNormal;
	if (fscanf(dataFile, "%d %d %d %d %d\n", &numGenes, &numSamples, &value, &numTumor, &numNormal) != 5) {
		printf("Error reading the first line numbers\n");
		fclose(dataFile);
		return 1; 
	}

	printf("%d %d %d %d %d\n", numGenes, numSamples, value, numTumor, numNormal); 

	std::vector<Gene> sparseData (numGenes);

	int fileRows = numGenes * numSamples;

	for (int i = 0; i < fileRows; i++){
		int gene, sample;	
		if (fscanf(dataFile, "%d %d %d %s %s\n", &gene, &sample, &value, geneId, sampleId) != 5) {
			printf("Error reading the line numbers\n");
			fclose(dataFile);
			return 1; 
		}
	
		if (value > 0){
			if (sample < numTumor){
				sparseData[gene].tumor.insert(sample);
			}
			else{
				sparseData[gene].normal.insert(sample);
			}
		}
	}

	auto start = std::chrono::high_resolution_clock::now();
	funcName(sparseData, numGenes, numTumor, numNormal);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
        printf("Time taken: %.3f ms\n", duration.count());
}
