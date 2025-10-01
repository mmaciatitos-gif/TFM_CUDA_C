#include <stdio.h>

__global__ void example(void){
}

int main(void){



example<<<2,1>>>();

return 0;

}

