package com.tfg.basico.repository;

import com.tfg.basico.model.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

/**
 * Interfaz para gestionar las consultas de la tabla "user".
 */
@Repository
public interface UserRepository extends JpaRepository<User, Integer> {
    
    // Busca los datos de un usuario en la base de datos a partir de su apodo de login
    User findByUsername(String username);
}