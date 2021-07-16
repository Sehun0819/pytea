matmul
    relu
        matmul
            reshape
                avgpool
                    relu
                        convol
                            relu
                                convol
                                    cipher tempCipher0:[10 1 28 28]
                                    plain conv1.weight:[32 1 3 3]
                                +
                                plain conv1.bias:[32]
                            plain conv2.weight:[64 32 3 3]
                        +
                        plain conv2.bias:[64]
                    2
                    2
                [10 9216]
            transpose
                plain fc1.weight:[128 9216]
                0
                1
        +
        plain fc1.bias:[128]
    transpose
        plain fc2.weight:[10 128]
        0
        1
+
plain fc2.bias:[10]