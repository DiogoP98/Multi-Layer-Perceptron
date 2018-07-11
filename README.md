# Neural Networks

### Descrição do programa:

Este programa é composto pelos seguintes ficheiros: ml.cpp. <br/> 
O programa permite testar redes neuronais de n inputs e 1 output, com uma hidden layer de tamanho variavel. Os exemplos utilizados pelo programa podem ser gerados no proprio programa ou passados como input ao correr o mesmo (por defeito está como gerar exemplos no propio programa). <br/>
Adicionalmente ao programa incluimos o codigo para gerar exemplos caso seja pretendido passa los como input. <br/>

-------------------------------------------

### Requerimentos:

Compilador g++ com c++ versão 11. <br/>

-------------------------------------------

### Instruções para compilar e executar:

Compilar: g++ -std=c++11 ml.cpp -o ml <br/> 
executar: ./ml <br/>

-------------------------------------------

### Execução:

No inicio do programa vai ser questionado o numero de inputs, o numero de nos na camada hidden, o numero de repeticoes para o programa correr e a learning rate. Em cada repeticao vai dar um output do genero k, epochs, em k é o numero da repetição. Caso nao convirga em vez de o numero de epchs tem uma linha. No final é apresentado o numero mé dio de epchs, o numeros de repeticoes que convergiram e o tempo medio que demora a fazer 1000 epochs.

-------------------------------------------

### Autores:

| Nome              | Numero UP     |
| ----------------- | ------------- |
| Diogo Pereira     | 201605323     |
| Afonso Fernandes  | 201606852     |
