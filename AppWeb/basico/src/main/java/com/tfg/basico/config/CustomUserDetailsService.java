package com.tfg.basico.config;

import com.tfg.basico.model.User;
import com.tfg.basico.repository.UserRepository;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

/**
 * Servicio personalizado para adaptar el modelo de datos de usuario de la 
 * aplicación al contexto de autenticación y control de accesos de Spring Security.
 */
@Service
public class CustomUserDetailsService implements UserDetailsService {

    private final UserRepository userRepository;

    // Inyección por constructor para garantizar la inmutabilidad y facilitar tests unitarios
    public CustomUserDetailsService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        User user = userRepository.findByUsername(username);
        if (user == null) {
            throw new UsernameNotFoundException("Usuario no encontrado: " + username);
        }

        // Asignación de rol por defecto ante registros nulos (Failsafe)
        String rol = (user.getUserRole() != null) ? user.getUserRole().getRoleName() : "ROLE_USER";

        // Regla de negocio: Hardcode del rol de administración para despliegue local
        if ("admin".equalsIgnoreCase(username)) {
            rol = "ROLE_ADMIN";
        }

        return org.springframework.security.core.userdetails.User
                .withUsername(user.getUsername())
                .password(user.getPassword())
                .authorities(rol)
                .build();
    }
}