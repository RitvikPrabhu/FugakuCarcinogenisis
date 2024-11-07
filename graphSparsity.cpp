#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <map>
#include <vector>
#include <set>
#include <algorithm>

#define MAX_BUF_SIZE 1024

struct Node {
	bool oneTumor;
	bool noNormal;
};


void funcName(std::vector<std::set<int> > data, int totalGenes, int numTumor, int numNormal){
	for (int i = 0; i < totalGenes; i++){
		std::set<int> gene1 = data[i];
		for (int j = i + 1; j < totalGenes; j++){
			std::set<int> gene2 = data[j];
			std::set<int> intersect;

			std::set_intersection(gene1.begin(), gene1.end(), gene2.begin(), gene2.end(),
                        std::inserter(intersect, intersect.begin()));
			
			for (int k = j + 1; k < totalGenes; k++){
				printf("Combination: %d %d %d\n", i, j, k);
			}
		
		}
	
	}

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

	std::vector<std::set<int> > sparseData (numGenes);

	int fileRows = numGenes * numSamples;

	for (int i = 0; i < fileRows; i++){
		int gene, sample;	
		if (fscanf(dataFile, "%d %d %d %s %s\n", &gene, &sample, &value, geneId, sampleId) != 5) {
			printf("Error reading the line numbers\n");
			fclose(dataFile);
			return 1; 
		}
		
		if (value > 0){
			sparseData[gene].insert(sample);
		}
	}

	funcName(sparseData, numGenes, numTumor, numNormal);



}
