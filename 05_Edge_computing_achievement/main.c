

#include "./SYSTEM/sys/sys.h"
#include "./SYSTEM/usart/usart.h"
#include "./SYSTEM/delay/delay.h"
#include "./BSP/LED/led.h"
#include "weights.h"
#include "nnom.h"
#include "stdio.h"
#include "data_array.c" 
volatile int8_t current_index = 0; 

#define SERIAL_DEBUG
#define QUANTIFICATION_SCALE 1
#define ZERO_POINT (-2)               // caculation of Z 
//#define QUANTIFICATION_SCALE 1
//#define ZERO_POINT 0
#define OUPUT_THRESHOLD 63 //The out put of model must bigger than this value unless the out put would be unrecognized.

#ifdef NNOM_USING_STATIC_MEMORY
	uint8_t static_buf[1024 *300];
#endif //NNOM_USING_STATIC_MEMORY
nnom_model_t* model;

typedef int8_t Model_Output_t;
volatile Model_Output_t model_output = -1;

#define INPUT_SIZE 2999

void model_feed_data(void) {
    uint16_t i = 0;
    const double scale = QUANTIFICATION_SCALE; //QUANTIFICATION_SCALE 

   
    for (i = 0; i < INPUT_SIZE; i++) {
        nnom_input_data[i] = data_array_5[current_index][i];;
    }

   printf("useing array: %d\n", current_index);
    current_index++;
    if (current_index >= 5) {
        current_index = 0; // 
    }
}

typedef enum eModel_Output{
	Unrecognized = -1,
	category_0= 0,
	category_1 = 1,
	category_2 = 2,
	category_3 = 3,
	category_4 = 4,
	category_5 = 5,
	category_6 = 6,
	category_7 = 7,
	category_8 = 8,


}eModel_Output;



Model_Output_t model_get_output(void)
{
	volatile uint8_t i = 0;
	volatile Model_Output_t max_output = -128;
	Model_Output_t ret = 0;
	model_feed_data();
	model_run(model);
	for(i = 0; i < 9;i++){
		
		#ifdef SERIAL_DEBUG
	
		printf("Output[%d] = %.2f %%\n",i,(nnom_output_data[i] / 127.0)*100);
		#endif //SERIAL_DEBUG
		
		if(nnom_output_data[i] >= max_output){
			max_output = nnom_output_data[i] ;
			ret = i;
		}
	}
	if(max_output < OUPUT_THRESHOLD || ret == Unrecognized){
		ret = Unrecognized;
	}
	int ret_int = (int)ret;  
	#ifdef SERIAL_DEBUG
	

	switch(ret_int){
		case Unrecognized:
			printf("Unrecognized");
			break;
		case category_0:
			printf("category_1");
			break;
		case category_1:
			printf("category_2");
			break;
		case category_2:
			printf("category_3");
			break;
		case category_3:
			printf("category_4");
			break;
		case category_4:
			printf("category_5");
			break;
		case category_5:
			printf("category_6");
			break;
		case category_6:
			printf("category_7");
			break;
		case category_7:
			printf("category_8");
			break;		
		case category_8:
			printf("category_9");
			break;
		
	}
	printf("\n");
	#endif //SERIAL_DEBUG
	
	return ret_int;
}


int main(void)
{
    sys_cache_enable();                  /* 打开L1-Cache */
    HAL_Init();                          /* 初始化HAL库 */
    sys_stm32_clock_init(240, 2, 2, 4);  /* 设置时钟, 480Mhz */
    delay_init(480);                     /* 延时初始化 */
    led_init();                          /* 初始化LED */
		usart_init(115200);
    #ifdef NNOM_USING_STATIC_MEMORY
		nnom_set_static_buf(static_buf, sizeof(static_buf)); 
	  #endif //NNOM_USING_STATIC_MEMORY
			  LED2(1);                         /* LED2(BLUE)  灭 */
        LED0(0);                         /* LED0(RED)   亮 */
        delay_ms(500);
	    printf("while1");
  			model = nnom_model_create();
	    printf("while");
	      LED2(0);                         /* LED2(BLUE)   */
        LED0(1);                         /* LED0(RED)    */
        delay_ms(500);

//		printf("While");
    while (1)
    {
//        LED2(1);                         /* LED2(BLUE)  灭 */
//        LED0(0);                         /* LED0(RED)   亮 */
//        delay_ms(500);
//        LED0(1);                         /* LED0(RED)   灭 */
//        LED1(0);                         /* LED1(GREEN) 亮 */
//        delay_ms(500);
//        LED1(1);                         /* LED1(GREEN) 灭 */
//        LED2(0);                         /* LED2(BLUE)  亮 */
        delay_ms(50);
     model_output = model_get_output();
    }
}
