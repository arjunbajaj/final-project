// |-------------------------------------|
// |       [CM3070] FINAL PROJECT        |
// |    CLASSIFYING SATELLITE IMAGERY    |
// |    BSc CS - UNIVERSITY OF LONDON    |
// |             ARJUN BAJAJ             |
// |-------------------------------------|

// Include the Pico standard lib and time libraries
#include <stdio.h>
#include "pico/stdlib.h"
#include "pico/time.h"

// Include the Tensorflow Libraries
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

// Include the Model File
#include "eurosat.h"

// The Image Size is 64x64 px image of 3 channels (RGB), in float32 format
// That is 49,152 bytes per image.
#define IMAGE_SIZE 64 * 64 * 3 * 4


/**
 * The `read_image` function reads the image from the UART over USB connection.
 *
 * The UART over USB is configured in CMakeLists.txt in the following lines:
 *      pico_enable_stdio_usb(diode 1)
 *      pico_enable_stdio_uart(diode 0)
 *
 * The function starts reading a single character (uint8) from the UART,
 * and waits for 100 microseconds. If a character arrives, the character is
 * added to a buffer. When the loop breaks, the buffer's length is checked.
 * If it is less than 10, which, I assume, happens because of Serial port chatter,
 * the buffer is dropped. Otherwise, the buffer's data is appended to the `image`
 * array. Until the image index reaches the desired length (49,152), the function
 * keeps waiting for the image to be sent.
 *
 * This code is based on code from Tony Smith:
 * https://blog.smittytone.net/2021/10/31/how-to-send-data-to-a-raspberry-pi-pico-via-usb/
 * however, my code is a heavily modified version of the original implementation,
 * and it is extremely specific to sending exactly one image of 49kb over UART.
 *
 * The Raspberry Pi docs for `getchar_timeout_us` are here:
 * https://raspberrypi.github.io/pico-sdk-doxygen/group__pico__stdio.html
 *
 * This kind of an implementation is required because when sending data over
 * Serial, I observed that the Operating System or Pico automatically chunk data
 * into anywhere from 256 to 384 bytes. If an OS does not chunk, the code below
 * will fail, because the `buffer[400]` will overflow.
 *
 * This implementation is tested on an M1 Macbook Air, and it works well.
 */
uint16_t read_image(uint8_t *image, uint8_t *buffer) {
  uint16_t image_index = 0;

  // Loop until an entire image is received
  while (image_index < IMAGE_SIZE) {
    // Temporary buffer to hold chunked data
    uint16_t buffer_index = 0;

    while (true) {
      // Get a character from UART
      int c = getchar_timeout_us(100);

      // If a character is received, write it to the buffer
      if (c != PICO_ERROR_TIMEOUT && buffer_index < 400) {
        buffer[buffer_index++] = (c & 0xFF);
      } else {
        break;
      }
    }

    if (buffer_index > 1) {
      // Drop buffer if it is incomplete.
      // Otherwise, copy the buffer data into the image.

      // Indicate data is being sent
      gpio_put(PICO_DEFAULT_LED_PIN, 1);

      // Copy buffer data into image
      for (uint16_t i = 0; i < buffer_index; i++) {
        image[image_index++] = buffer[i];
      }
    }
  }

  // The final image index is the length of the image
  return image_index;
}


int main(int argc, char* argv[]) {

  stdio_init_all();
  gpio_init(PICO_DEFAULT_LED_PIN);
  gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);

  uint8_t buffer[400] { 0 };
  uint8_t image[IMAGE_SIZE] { 0 };
  uint8_t category = 0;
  float probability_value = 0.0f;

  sleep_ms(1000);

  printf("\n");
  printf("\n\n\n\n");

  printf("-----------------------------------------------------------------------------\n");
  printf("  Satellite Image Classification using Tensorflow Lite on Raspberry Pi Pico  \n");
  printf("     CM3070 Final Project - University of London BSc Computer Science\n");
  printf("                            by Arjun Bajaj\n");
  printf("-----------------------------------------------------------------------------\n\n");

  printf("MESSAGE: Initializing Tensorflow Lite: ");
  gpio_put(PICO_DEFAULT_LED_PIN, 1);

  tflite::MicroErrorReporter micro_error_reporter;
  const tflite::Model* model = ::tflite::GetModel(eurosat_tflite);
  printf("Model Loaded, ");

  // Load The Tensorflow Ops required by the model.
  // Note: The examples show to use the AllOpsResolver,
  // however that ends up creating a large binary (around ~800kb).
  // With this method of only selecting the Ops we're using in the model
  // the size of the binary can be significantly reduced to around half!
  // It also allows more breathing room for the model's tensors in RAM.
  static tflite::MicroMutableOpResolver<10> micro_op_resolver;
  micro_op_resolver.AddReduceMax();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddDequantize();
  micro_op_resolver.AddMean();

  // The Tensor Arena Size is to be established by experimentation according
  // to the TensorflowLite docs. However, a minimum required arena size is
  // reported on startup if it falls short of what the model requires.
  // I have established with the current model, I need at least 158kb of
  // RAM to hold the tensors. This size includes the input tensor,
  // output tensor, and the model's own weights tensor.
  constexpr int kTensorArenaSize = 160 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];

  // Initialize the TensorflowLite Interpreter
  tflite::MicroInterpreter interpreter(
    model, micro_op_resolver, tensor_arena,
    kTensorArenaSize, &micro_error_reporter
  );

  // Allocate Tensors
  interpreter.AllocateTensors();
  printf("Tensors Allocated, ");

  // Get pointers to the input and output tensors.
  TfLiteTensor* tf_input = interpreter.input(0);
  TfLiteTensor* tf_output = interpreter.output(0);

  printf("TFLite Initialized.\n");
  gpio_put(PICO_DEFAULT_LED_PIN, 0);

  // Tensorflow Lite is now ready for inference.
  // We can wait for images to be sent over UART/USB now.
  // Sending an image can be as simple as:
  //       cat pico-test-images/0000_6.bin > /dev/<PICO_DEVICE>
  while (true) {
    // The host script looks for a "READY:" before sending data.
    printf("\nREADY: (send image over USB/UART Serial)\n");

    // Wait for an image to be sent over the UART/USB connection
    uint16_t image_length = read_image(image, buffer);

    printf("MESSAGE: Running Prediction\n");

    // Log time before inferencing
    auto start_time = to_ms_since_boot(get_absolute_time());

    // Copy image bytes over to the allocated input tensor
    memcpy(tf_input->data.f, image, tf_input->bytes);

    // Invoke Tensorflow Inferencing
    TfLiteStatus invoke_status = interpreter.Invoke();

    // If the invoke failed, print a message.
    // TFLite also prints a debug log which is really helpful.
    if (invoke_status != kTfLiteOk) {
      printf("MESSAGE: Tensorflow Lite Model Invoke Failed.\n");
    }

    // Calculate time taken for inference
    auto end_time = to_ms_since_boot(get_absolute_time());
    auto time_taken = end_time - start_time;

    // Read and print results
    category = 0;
    probability_value = 0.0f;

    printf("VALUES: ");
    // Find the value with the highest confidence,
    // and set that as the category predicted.
    // Essentially `argmax()`
    for (uint8_t i = 0; i < 10; i++) {
      if (tf_output->data.f[i] > probability_value) {
        category = i;
        probability_value = tf_output->data.f[i];
      }

      // Print all probability values to the console
      printf("%f,", i, tf_output->data.f[i]);
    }

    // Print the category predicted and time taken.
    printf("\nRESULT: %d\n", category);
    printf("TIME: %d ms\n", time_taken);

    // Switch off the LED, indicating all work is complete.
    // The LED is switched on again when an image is being received
    // in the `read_image()` function.
    gpio_put(PICO_DEFAULT_LED_PIN, 0);
  }

  return 0;
}
