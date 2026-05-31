package com.sistemasdistr.basico;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.web.client.RestTemplate;
import jakarta.servlet.MultipartConfigElement;

@SpringBootApplication
public class BasicoApplication {

    public static void main(String[] args) {
        SpringApplication.run(BasicoApplication.class, args);
    }

    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }

    @Bean
    public MultipartConfigElement multipartConfigElement() {
        long maxFileSize = 52428800L; // 50MB
        long maxRequestSize = 524288000L; // 500MB
        return new MultipartConfigElement(null, maxFileSize, maxRequestSize, 0);
    }
}