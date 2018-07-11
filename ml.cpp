#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <list>
#include <numeric>
#include <cmath>
#include <random>
#include <algorithm>

using namespace std;

//Magic numbers

#define titulo "Neural Networks"

int numberInputs = 4;
#define numberOutput 1
int numberHidden = 4;
#define numberLayers 2

//srand param
#define seed_i 1
#define factor_seed 1

#define limit 300000
int  repeticoes = 10;

#define cicloBatch   200
#define cicloTodo    1000
#define randperiodo  1

#define factor 2
#define factor2 2

#define withbias true
#define printNetwork false

#define learning_rate 0.35

//global variables

vector<vector<double> > in;
vector<double> out;

vector<int> nodesPerLayer;

vector< vector< vector< double> > > weights;
vector< vector< double> > values;
vector< vector< double> > bias;
vector< vector< double> > as;
vector< vector< double> > errors;


long sum;
long conv;
double maxerror;
double learningRate = learning_rate;
int numberExamples;
int  tamanhoBatch;

//reads input from terminal
void read_input(){
	int x;
	for(int i=0;i<numberExamples;i++){
		in.push_back(vector<double>());
		for(int j = 0;j<numberInputs;j++){	
			scanf("%d",&x);
			in[i].push_back(x);
		}
		scanf("%d",&x);
		out.push_back(x);
	}
	printf("\n");
}

void read_params(){
	printf("\n");
	int x;double xx;
	
	printf("Number of input nodes: ");
	scanf("%d",&x);
	numberInputs = x;
	
	printf("Number of hidden nodes: ");
	scanf("%d",&x);
	numberHidden = x;
	
	printf("Number of repetitions: ");
	scanf("%d",&x);
	repeticoes = x;
	
	printf("learning rate: ");
	scanf("%lf",&xx);
	learningRate = xx;
	
	printf("\n");
}

void generate_input(){
	list<pair<string,int> > v;
	v.push_back({"",0});

	string s;
	int k;
	while((int)v.front().first.size() != numberInputs){
		s = v.front().first;
		k = v.front().second;
		v.pop_front();
		v.push_back({s+"0",k});
		v.push_back({s+"1",k+1});
	}

	list<pair<string,int> >::iterator it;
	for(it = v.begin();it!=v.end();it++){
		in.push_back(vector<double>());
		for(int i=0;i<numberInputs;i++){
			in.back().push_back(double((*it).first[i]-'0'));
		}
		if((*it).second%2) out.push_back(1);
		else out.push_back(0);
	}
}

//intialize vectors
void init_vectors(int p){
	srand(p);
	double k;
	weights.push_back(vector<vector<double> >());
	for(int layer=1;layer<numberLayers+1;layer++){
		weights.push_back(vector<vector<double> >());
		for(int i=0;i<nodesPerLayer[layer];i++){
			weights[layer].push_back(vector<double>());
			for(int j=0;j<nodesPerLayer[layer-1];j++){
				k = (double)rand() / RAND_MAX;
				weights[layer][i].push_back(-1 + k*2);
			}
		}
	}
	
	for(int layer=0;layer<numberLayers+1;layer++){
		values.push_back(vector<double>());
		as.push_back(vector<double>());
		errors.push_back(vector<double>());
		bias.push_back(vector<double>());
		for(int i=0;i<nodesPerLayer[layer];i++){
			values[layer].push_back(0);
			as[layer].push_back(0);
			errors[layer].push_back(0);
			
			k = (double)rand() / RAND_MAX;
			if(withbias)bias[layer].push_back(-1 + k*2);
		}
	}
}

//clear vectors
void clear(){
	weights.clear();
	values.clear();
	as.clear();
	errors.clear();
	bias.clear();
}

//activation function
double actv(double x){
	return 1/(1+exp(-x));
}

//derivative
double derv_actv(double x){
	return x*(1-x);
}

//foward propagation
void foward(int ex){
	for(int i=0;i<numberInputs;i++){
		values[0][i] = in[ex][i];
		as[0][i] = values[0][i];
	}
	for(int layer = 1;layer<numberLayers+1;layer++){
		for(int i=0;i<nodesPerLayer[layer];i++){
			if(withbias) values[layer][i] = bias[layer][i];
			else values[layer][i] = 0;
			for(int j=0;j<nodesPerLayer[layer-1];j++){
				values[layer][i] += as[layer-1][j] * weights[layer][i][j];
			}
			as[layer][i] = actv(values[layer][i]);
		}
	}
}

//backward error propagation
void backward(int ex){
	for(int i=0;i<numberOutput;i++){
		errors[numberLayers][i] = (-as[numberLayers][i]+out[ex]) * derv_actv(as[numberLayers][i]);
		if(abs(out[ex]-as[numberLayers][0])>maxerror)maxerror = abs(out[ex]-as[numberLayers][0]);
	}
	double err;
	for(int layer=numberLayers-1;layer>=0;layer--){
		for(int i=0;i<nodesPerLayer[layer];i++){
			err = 0;
			for(int j=0;j<nodesPerLayer[layer+1];j++){
				err += errors[layer+1][j] * weights[layer+1][j][i];
			}
			errors[layer][i] = err * derv_actv(as[layer][i]);
		}
	}
}

//weight update
void update_weights(){
	for(int layer = 1;layer<numberLayers+1;layer++){
		for(int i=0;i<nodesPerLayer[layer];i++){
			for(int j = 0;j<nodesPerLayer[layer-1];j++){
				weights[layer][i][j] += learningRate * as[layer-1][j] * errors[layer][i];
			}
			if(withbias) bias[layer][i] += learningRate * errors[layer][i];
		}
	}
}

//print network
void print_Network(){
	for(int ex=0;ex<numberExamples;ex++){
		foward(ex);
		printf("(%02d,%.0lf) %lf %lf\n",ex,out[ex],as[numberLayers][0],abs(out[ex]-as[numberLayers][0]));
	}
	for(int layer = 1;layer<numberLayers+1;layer++){
		printf("Layer: %d\n",layer);
		for(int node = 0;node<nodesPerLayer[layer];node++){
			printf("\t(node: %d)\n",node);
			for(int node2 = 0;node2<nodesPerLayer[layer-1];node2++){
				printf("\t(%d) weight: %lf\n",node2,weights[layer][node][node2]);
			}
		}
	}
}

//train network
bool train(int epochs,int k){
	vector<int> pos;
	for(int i=0;i<numberExamples;i++) pos.push_back(i);
	int t = 0;
	bool b = true;
	int ii=0;
	maxerror =-1;
	double prevmaxerror =-100;
	
	while(b){
		//randomize examples
		if( t> ii*randperiodo){
			random_shuffle(pos.begin(),pos.end());
			ii++;
		}
		//train in each batch a max numebr times and then in all examples
		for(int i=0;i<ceil(numberExamples/tamanhoBatch);i++){
			int t2=0;
			while(b){
				b = false;
				for(int ex =tamanhoBatch*i;ex<tamanhoBatch*(i+1) && ex < numberExamples;ex++){
					foward(pos[ex]);
					backward(pos[ex]);
					if(abs(out[ex]-as[numberLayers][0])>prevmaxerror/factor2)update_weights();
					if(abs(out[ex]-as[numberLayers][0])>0.05 && abs(out[ex]-as[numberLayers][0])>prevmaxerror/factor2){
						b=true;
					}
				}
				if(t2>=(int)cicloBatch/ceil(numberExamples/tamanhoBatch)) break;
				t2++;
				t++;
				if(maxerror!=-1)prevmaxerror=maxerror;
				else prevmaxerror =-100;
				maxerror = -1;
			}
			
			t2=0;
			while(b){
				b = false;
				for(int ex =0;ex < numberExamples;ex++){
					foward(ex);
					backward(ex);
					if(abs(out[ex]-as[numberLayers][0])>prevmaxerror/factor)update_weights();
					if(abs(out[ex]-as[numberLayers][0])>0.05 && abs(out[ex]-as[numberLayers][0])>prevmaxerror/factor){
						b=true;
					}
				}
				if(t2>=(int)cicloTodo/ceil(numberExamples/tamanhoBatch)) break;
				t2++;
				t++;
				if(maxerror!=-1)prevmaxerror=maxerror;
				else prevmaxerror =-100;
				maxerror = -1;
			}
		}
			
		//see if complete
		b = false;
		for(int ex =0;ex<numberExamples;ex++){
			foward(ex);
			backward(ex);
			if(abs(out[ex]-as[numberLayers][0])>prevmaxerror/factor)update_weights();
			if(abs(out[ex]-as[numberLayers][0])>0.05 && abs(out[ex]-as[numberLayers][0])>prevmaxerror/factor){
				//printf("(%03d,%.0lf) %lf %lf\n",ex,out[ex],as[numberLayers][0],abs(out[ex]-as[numberLayers][0]));
				b=true;
			}
		}
		//printf("%d\n\n",t);
		if(maxerror!=-1)prevmaxerror=maxerror;
		else prevmaxerror =-100;
		maxerror = -1;
		t++;
    
		if(t>=epochs+1) break;
	}
	//if complete b is false
	if(!b){
		conv++;
		sum += t;
		printf("%d\t%d\n",k,t);
		
		if(printNetwork) print_Network();
	}
	else{
		//failed
		printf("%d\t--------\n",k);
	}
	return b;
}

int main(){
	printf("%s\n",titulo);
	
	//read_input();
	
	read_params();
	generate_input();
	
	numberExamples = (int)pow(2,numberInputs);
	tamanhoBatch = (int)(0.75*numberExamples);
	nodesPerLayer = {numberInputs,numberHidden,numberOutput};
	
	sum = 0;
	conv = 0;
	long start_s,stop_s;
	double time = 0;
	int prev = 0;
	for(int k = seed_i;k<repeticoes+seed_i;k++){
		init_vectors((k+1)*(factor_seed+1));
		start_s=clock();
		if(!train(limit,k-seed_i));
		stop_s=clock();
		if(conv!=prev) time+=(stop_s-start_s)/double(CLOCKS_PER_SEC)*1000;
		prev = conv;
		clear();
	}
	printf("\n\n");
	printf("Average:\nlearningrate = %.2lf\nnumber epochs = %.1lf\number successful = %ld\n",learningRate,sum*1.0/conv,conv);
	printf("time per 1000 epochs: %lf ms\n\n",time/sum*1000);
	return 0;
}
