#include <stdio.h>
#include <list>
#include <string>

#define in 3

using namespace std;

int main(){
	list<pair<string,int> > v;
	v.push_back({"",0});

	string s;
	int k;
	while(v.front().first.size() != in*2){
		s = v.front().first;
		k = v.front().second;
		v.pop_front();
		v.push_back({s+"0 ",k});
		v.push_back({s+"1 ",k+1});
	}

	list<pair<string,int> >::iterator it;
	for(it = v.begin();it!=v.end();it++){
		printf("%s",(*it).first.c_str());
		if((*it).second%2) printf("1\n");
		else printf("0\n");
	}
}
