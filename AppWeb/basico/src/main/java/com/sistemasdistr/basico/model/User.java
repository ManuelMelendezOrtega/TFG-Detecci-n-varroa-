package com.sistemasdistr.basico.model;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import java.io.Serializable;
import java.time.LocalDateTime;

@Entity
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Table(name = "user")
public class User implements Serializable {
    @Id
    @Column(name = "id")
    @GeneratedValue(strategy = GenerationType.AUTO)
    private Integer id;
    @Column(name = "username", length = 50)
    private String username;
    @Column(name = "email", length = 50)
    private String email;
    @Column(name = "nombre_usuario", length = 30)
    private String nombreUsuario;
    @Column(name = "password", length = 250)
    private String password;

    @Lob
    private byte[] publickey;

    @Column(name = "fechaUltimoAcceso")
    private LocalDateTime fechaUltimoAcceso;

    @ManyToOne(fetch = FetchType.EAGER)
    private Role userRole;

}