package com.tfg.basico.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.io.Serializable;

/**
 * Entidad que representa la tabla "role" en la base de datos.
 * Define los distintos niveles de permisos (ej. Administrador o Apicultor normal) 
 * que puede tener un usuario dentro de la plataforma.
 */
@Entity
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Table(name = "role")
public class Role implements Serializable {

    // Identificador único del rol
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    // Nombre oficial del permiso (Normalmente "ROLE_USER" o "ROLE_ADMIN")
    @Column(nullable = false)
    private String roleName;

    // Interruptor de seguridad (1 = visible, 0 = oculto). 
    // Sirve para evitar que alguien desde la pantalla de registro público 
    // pueda crearse una cuenta de Administrador por accidente o de forma maliciosa.
    @Column(nullable = false)
    private Integer showOnCreate;
}