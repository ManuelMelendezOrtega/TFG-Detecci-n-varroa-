package com.tfg.basico.repository;

import com.tfg.basico.model.Role;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

/**
 * Interfaz para gestionar las consultas de la tabla "role".
 */
@Repository
public interface RoleRepository extends JpaRepository<Role, Integer> {

    // Busca un rol en la base de datos introduciendo su nombre (ej. "ROLE_USER")
    Role findByRoleName(String roleName);
}