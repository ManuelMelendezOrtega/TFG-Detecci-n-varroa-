package com.tfg.basico.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.crypto.factory.PasswordEncoderFactories;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;

/**
 * Clase de configuración global de Spring Security.
 * Define las políticas de acceso (RBAC - Role Based Access Control), 
 * el enrutamiento de seguridad y los algoritmos de encriptación.
 */
@Configuration
public class SecurityConfig {

    private final CustomUserDetailsService customUserDetailsService;

    public SecurityConfig(CustomUserDetailsService customUserDetailsService) {
        this.customUserDetailsService = customUserDetailsService;
    }

    /**
     * Define el algoritmo de hash para las contraseñas.
     * Utiliza el DelegatingPasswordEncoder para adaptarse dinámicamente a 
     * algoritmos modernos (como BCrypt) garantizando la escalabilidad de la seguridad.
     */
    @Bean
    public PasswordEncoder passwordEncoder() {
        return PasswordEncoderFactories.createDelegatingPasswordEncoder();
    }

    @Bean
    public AuthenticationManager authenticationManager(AuthenticationConfiguration config) throws Exception {
        return config.getAuthenticationManager();
    }

    /**
     * Configuración del filtro de seguridad HTTP (Pipeline de Spring Security).
     * Establece qué rutas son públicas, cuáles requieren autenticación y gestiona
     * los flujos de inicio y cierre de sesión.
     */
    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            .csrf(AbstractHttpConfigurer::disable) // Deshabilitado para simplificar peticiones POST locales/API
            .userDetailsService(customUserDetailsService)
            .authorizeHttpRequests(auth -> auth
                // Rutas públicas: estáticos, recursos de la IA y pantallas de entrada
                .requestMatchers("/", "/login", "/registro", "/css/**", "/js/**", "/detections/**").permitAll()
                // Rutas protegidas por Rol
                .requestMatchers("/admin/**").hasRole("ADMIN")
                // Cualquier otra ruta requiere estar logueado
                .anyRequest().authenticated() 
            )
            .formLogin(form -> form
                .loginPage("/login")          
                .defaultSuccessUrl("/", true)  
                .permitAll()
            )
            .logout(logout -> logout
                .logoutSuccessUrl("/login?logout")
                .permitAll()
            );

        return http.build();
    }
}