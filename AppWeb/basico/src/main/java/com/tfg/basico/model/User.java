package com.tfg.basico.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * Entidad que representa la tabla "user" en la base de datos.
 * Almacena los datos de perfil, las credenciales de acceso y el nivel de permisos
 * de los apicultores y administradores del sistema.
 */
@Entity
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Table(name = "user")
public class User implements Serializable {

    // Identificador único (Clave Primaria) generado automáticamente por el sistema
    @Id
    @Column(name = "id")
    @GeneratedValue(strategy = GenerationType.AUTO)
    private Integer id;

    // Nombre único utilizado para iniciar sesión en la plataforma (ej. "juan92")
    @Column(name = "username", length = 50)
    private String username;

    // Correo electrónico de contacto del apicultor
    @Column(name = "email", length = 50)
    private String email;

    // Nombre real o completo del usuario para mostrarlo en los saludos de la interfaz
    @Column(name = "nombre_usuario", length = 30)
    private String nombreUsuario;

    // Contraseña de acceso (Se le asigna un tamaño de 250 porque se guarda encriptada)
    @Column(name = "password", length = 250)
    private String password;

    // Campo especial para almacenar bloques de datos grandes (como una clave criptográfica pública)
    @Lob
    private byte[] publickey;

    // Registra de forma automática el día y la hora exactos del último inicio de sesión
    @Column(name = "fechaUltimoAcceso")
    private LocalDateTime fechaUltimoAcceso;

    // Relación Many-To-One: Varios usuarios pueden compartir el mismo Rol (ej. muchos usuarios "USER").
    // Se usa FetchType.EAGER para cargar el rol inmediatamente junto con los datos del usuario.
    @ManyToOne(fetch = FetchType.EAGER)
    private Role userRole;

}