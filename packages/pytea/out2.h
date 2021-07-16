matmul
    relu
        matmul
            reshape
                maxpool
                    relu
                        convol
                            relu
                                convol
                                    cipher tempCipher0:[10 1 28 28]
                                    plain conv1weight:[32 1 3 3]
                                +
                                plain conv1bias:[32]
                            plain conv2weight:[64 32 3 3]
                        +
                        plain conv2bias:[64]
                    2
                    2
                [10 9216]
            plain fc1weight:[128 9216]
        +
        plain fc1bias:[128]
    plain fc2weight:[10 128]
+
plain fc2bias:[10]
