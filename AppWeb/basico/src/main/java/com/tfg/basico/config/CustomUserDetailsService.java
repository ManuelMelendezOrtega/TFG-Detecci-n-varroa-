package com.tfg.basico.config;

import com.tfg.basico.model.User;
import com.tfg.basico.repository.UserRepository;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

@Service
public class CustomUserDetailsService implements UserDetailsService {

    private final UserRepository userRepository;

    public CustomUserDetailsService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        User user = userRepository.findByUsername(username);
        if (user == null) throw new UsernameNotFoundException("Usuario no encontrado");

        String rol = (user.getUserRole() != null) ? user.getUserRole().getRoleName() : "ROLE_USER";

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